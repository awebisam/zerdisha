"""Core orchestrator that manages CA, PD, and MA agents.

Behavioral guarantees:
- Pattern Detector (PD) runs in the background and never blocks CA.
- PD is scheduled every N interactions (default 5) and after artifact commands.
- Metacognitive Agent (MA) is command-only; no automatic analysis or persona updates during normal turns.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from ..models.graph import Session, Node, Edge, NodeType, EdgeType, Vector
from ..models.config import Settings
from ..database.neo4j_client import Neo4jClient
from ..database.mongodb_client import MongoDBClient
from ..agents.conversational import ConversationalAgent
from ..agents.pattern_detector import PatternDetector
from ..agents.metacognitive import MetacognitiveAgent
from ..core.embeddings import EmbeddingService
from ..core.analytics import AnalyticsEngine

logger = logging.getLogger(__name__)


class ExplorationEngine:
    """Main orchestrator for Zerdisha."""

    def __init__(self, settings: Settings):
        self.settings = settings

        # Initialize databases
        self.graph_db = Neo4jClient(settings.database_config)
        self.message_db = MongoDBClient(
            settings.mongodb_uri, settings.mongodb_database)

        # Initialize agents
        self.ca = ConversationalAgent(
            settings.llm_config, settings.persona_config)
        self.pd = PatternDetector(settings.llm_config)
        self.ma = MetacognitiveAgent(settings.llm_config)

        # Initialize services
        self.embedding_service = EmbeddingService(settings.llm_config)
        self.analytics = AnalyticsEngine(self.graph_db, self.embedding_service)

        # Current session
        self.current_session: Optional[Session] = None

        # PD background processing (initialized in initialize())
        self._pd_queue: Optional[asyncio.Queue] = None
        self._pd_worker_task: Optional[asyncio.Task] = None
        self._pd_cadence: int = 5  # run PD every 5 exchanges

        # MA last analysis cache (command-only)
        self._last_ma_analysis: Optional[Dict[str, Any]] = None

    async def initialize(self) -> None:
        """Initialize the engine and all components."""
        # Connect to databases
        self.graph_db.connect()
        self.graph_db.create_indexes()
        await self.message_db.connect()

        # Initialize agents
        await self.ca.initialize()
        await self.pd.initialize()
        await self.ma.initialize()

        # Start PD background worker
        self._pd_queue = asyncio.Queue()
        self._pd_worker_task = asyncio.create_task(self._pd_worker_loop())

        logger.info("Exploration Engine initialized")

    async def shutdown(self) -> None:
        """Shutdown the engine cleanly."""
        if self.current_session and self.current_session.status == "active":
            await self.end_session()

        # Stop PD worker
        if self._pd_worker_task:
            self._pd_worker_task.cancel()
            try:
                await self._pd_worker_task
            except asyncio.CancelledError:
                pass

        self.graph_db.close()
        await self.message_db.close()
        logger.info("Exploration Engine shutdown")

    async def start_session(self, topic: str, title: Optional[str] = None) -> Session:
        """Start a new exploration session."""
        if self.current_session and self.current_session.status == "active":
            await self.end_session()

        session_id = str(uuid.uuid4())
        session_title = title or f"Exploring {topic}"

        self.current_session = Session(
            id=session_id,
            title=session_title,
            topic=topic,
            start_time=datetime.utcnow(),
            status="active"
        )

        # Create session in MongoDB (full data) and Neo4j (summary)
        mongo_success = await self.message_db.create_session({
            "id": self.current_session.id,
            "title": self.current_session.title,
            "topic": self.current_session.topic,
            "start_time": self.current_session.start_time,
            "status": self.current_session.status,
            "messages": []
        })

        neo4j_success = self.graph_db.create_session_summary(
            self.current_session)

        if not (mongo_success and neo4j_success):
            raise RuntimeError(f"Failed to create session in databases")

        # Load relevant past nodes for context
        relevant_nodes = await self._load_relevant_context(topic)

        # Initialize CA with topic and context
        await self.ca.start_session(topic, relevant_nodes)

        logger.info(f"Started session: {session_id} - {session_title}")
        return self.current_session

    async def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input through CA; schedule PD in background; MA is command-only."""
        if not self.current_session:
            raise RuntimeError("No active session. Start a session first.")

        # Get response from Conversational Agent
        ca_response = await self.ca.process_input(user_input)

        # Store message exchange in MongoDB
        await self.message_db.add_message_exchange(
            self.current_session.id,
            user_input,
            ca_response["message"],
            metadata={
                "ca_reasoning": ca_response.get("reasoning", {}),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        # Keep in-memory messages for current session processing
        message_pair = {
            "timestamp": datetime.utcnow().isoformat(),
            "user": user_input,
            "assistant": ca_response["message"]
        }
        self.current_session.messages.append(message_pair)

        # Schedule PD in background on cadence (every self._pd_cadence exchanges)
        try:
            if self._pd_queue is not None and self.current_session:
                exchange_count = len(self.current_session.messages)
                should_schedule = (exchange_count % self._pd_cadence == 0)
                if should_schedule:
                    await self._pd_queue.put({
                        "type": "exchange",
                        "session_id": self.current_session.id,
                        "user_input": user_input,
                        "assistant": ca_response["message"],
                        "context": self.current_session.messages[-5:]
                    })
        except Exception as e:
            logger.error(f"Failed to enqueue PD task: {e}")

        # Update MongoDB with analysis data
        analysis_data = {
            # MA is command-only now; leave insights/flags empty unless analyze command is run
            "ma_insights": [],
            "concepts_extracted": 0,  # PD runs async; count unknown at this moment
            "ma_flags": [],
            "total_exchanges": len(self.current_session.messages)
        }
        await self.message_db.update_session_analysis(self.current_session.id, analysis_data)

        return {
            "message": ca_response["message"],
            "session_id": self.current_session.id,
            # PD runs in background; report zero immediately
            "new_concepts": 0,
            # MA insights only available via analyze command
            "ma_insights": [],
            "suggested_commands": ["/map", "/gapcheck", "/analyze", "/seed", "/end"]
        }

    async def execute_command(self, command: str, args: List[str] = None) -> Dict[str, Any]:
        """Execute TUI commands like /map, /gapcheck, /seed."""
        args = args or []

        if command == "map":
            result = await self._show_session_map()
            # Schedule PD consolidation after artifact generation
            await self._enqueue_pd_consolidate()
            return result
        elif command == "gapcheck":
            result = await self._gap_check()
            await self._enqueue_pd_consolidate()
            return result
        elif command == "seed":
            result = await self._inject_seed()
            await self._enqueue_pd_consolidate()
            return result
        elif command == "analyze":
            return await self._command_analyze(apply=False)
        elif command in ("apply_ma", "apply-ma"):
            return await self._command_apply_ma()
        elif command == "end":
            return await self.end_session()
        else:
            return {"error": f"Unknown command: {command}"}

    async def end_session(self) -> Dict[str, Any]:
        """End the current session."""
        if not self.current_session:
            return {"message": "No active session to end"}

        self.current_session.end_time = datetime.utcnow()
        self.current_session.status = "completed"

        # Final session analysis (command initiated)
        final_metrics = await self.ma.finalize_session(self.current_session)

        # End session in MongoDB with final analysis
        await self.message_db.end_session(self.current_session.id, final_metrics)

        session_summary = {
            "session_id": self.current_session.id,
            "duration_minutes": (self.current_session.end_time - self.current_session.start_time).total_seconds() / 60,
            "total_exchanges": len(self.current_session.messages),
            "concepts_created": len(self.current_session.nodes_created),
            "connections_made": len(self.current_session.edges_created),
            "final_metrics": final_metrics
        }

        # Schedule final PD consolidation
        await self._enqueue_pd_consolidate()

        logger.info(f"Ended session: {self.current_session.id}")
        self.current_session = None

        return session_summary

    async def review_session(self, session_date: str) -> Optional[Dict[str, Any]]:
        """Review a past session by date."""
        return await self.analytics.review_session(session_date)

    async def _load_relevant_context(self, topic: str) -> List[Dict[str, Any]]:
        """Load relevant nodes from past sessions for context."""
        relevant_nodes = self.graph_db.search_nodes_by_content(topic, limit=10)
        return relevant_nodes

    async def _show_session_map(self) -> Dict[str, Any]:
        """Show current session's concept map with comprehensive fallback mechanisms for missing data."""
        if not self.current_session:
            return {
                "error": "No active session",
                "message": "🗺️ **Start a session to see your concept map**\n\nThe session map shows concepts and connections you've explored.\n\n*Begin exploring a topic, then use `/map` to visualize your journey.*",
                "recovery_suggestions": ["Start a new session", "Engage in conversation to create concepts", "Try map after discussing ideas"]
            }

        # Initialize collections with error tracking
        nodes = []
        edges = []
        missing_nodes = []
        missing_edges = []
        node_errors = []
        edge_errors = []

        # Performance optimization: batch fetch nodes and edges in single database operation
        logger.debug(
            f"Batch fetching {len(self.current_session.nodes_created)} nodes and {len(self.current_session.edges_created)} edges for session map")

        try:
            batch_result = self.graph_db.get_session_nodes_and_edges_batch(
                self.current_session.nodes_created,
                self.current_session.edges_created
            )
            # Some mocks may return MagicMock; validate structure
            if isinstance(batch_result, dict):
                nodes = batch_result.get("nodes", [])
                edges = batch_result.get("edges", [])
            else:
                raise RuntimeError("Invalid batch result type")

            # Check for missing data
            fetched_node_ids = {node.get("id") for node in nodes}
            fetched_edge_ids = {edge.get("id") for edge in edges}

            missing_nodes = [
                nid for nid in self.current_session.nodes_created if nid not in fetched_node_ids]
            missing_edges = [
                eid for eid in self.current_session.edges_created if eid not in fetched_edge_ids]

            if missing_nodes:
                logger.warning(
                    f"Missing {len(missing_nodes)} nodes from batch fetch")
            if missing_edges:
                logger.warning(
                    f"Missing {len(missing_edges)} edges from batch fetch")

        except Exception as batch_error:
            logger.error(
                f"Batch fetch failed, falling back to individual queries: {batch_error}")

            # Fallback to individual queries if batch fails
            for node_id in self.current_session.nodes_created:
                try:
                    node_data = self.graph_db.get_node(node_id)
                    if node_data:
                        nodes.append(node_data)
                    else:
                        missing_nodes.append(node_id)
                        logger.warning(f"Node {node_id} not found in database")
                except Exception as e:
                    node_errors.append({"node_id": node_id, "error": str(e)})
                    logger.error(f"Error fetching node {node_id}: {e}")

            for edge_id in self.current_session.edges_created:
                try:
                    edge_data = self.graph_db.get_edge(edge_id)
                    if edge_data:
                        edges.append(edge_data)
                    else:
                        missing_edges.append(edge_id)
                        logger.warning(f"Edge {edge_id} not found in database")
                except Exception as e:
                    edge_errors.append({"edge_id": edge_id, "error": str(e)})
                    logger.error(f"Error fetching edge {edge_id}: {e}")

        # Handle case where no data is available
        if not nodes and not edges:
            if self.current_session.nodes_created or self.current_session.edges_created:
                # Session claims to have data but we can't fetch it
                return {
                    "error": "Session data unavailable",
                    "message": "🔧 **Session map temporarily unavailable**\n\nYour exploration data exists but can't be displayed right now. This might be a database connectivity issue.\n\n*Continue exploring - your progress is being saved. Try the map again in a moment.*",
                    "recovery_suggestions": ["Try map again in a moment", "Continue exploring", "Check database connectivity"],
                    "session_id": self.current_session.id,
                    "topic": self.current_session.topic,
                    "nodes": [],
                    "edges": [],
                    "node_count": 0,
                    "connection_count": 0,
                    "data_issues": {
                        "missing_nodes": len(missing_nodes),
                        "missing_edges": len(missing_edges),
                        "node_errors": len(node_errors),
                        "edge_errors": len(edge_errors)
                    }
                }
            else:
                # Session genuinely has no data yet
                return {
                    "message": "🌱 **Your exploration is just beginning**\n\nNo concepts mapped yet! As you discuss ideas and share your thoughts, I'll create a visual map of your exploration.\n\n*Keep exploring - concepts and connections will appear as we go.*",
                    "session_id": self.current_session.id,
                    "topic": self.current_session.topic,
                    "nodes": [],
                    "edges": [],
                    "node_count": 0,
                    "connection_count": 0,
                    "status": "empty_session"
                }

        # Generate contextual relationship descriptions with error handling
        relationship_descriptions = []
        if edges:
            try:
                relationship_descriptions = await self._generate_relationship_descriptions(nodes, edges)
            except Exception as desc_error:
                logger.error(
                    f"Failed to generate relationship descriptions: {desc_error}")
                # Create fallback descriptions
                relationship_descriptions = self._create_fallback_relationship_descriptions(
                    nodes, edges)

        # Prepare warnings for missing data
        warnings = []
        if missing_nodes:
            warnings.append(
                f"{len(missing_nodes)} concepts couldn't be displayed")
        if missing_edges:
            warnings.append(
                f"{len(missing_edges)} connections couldn't be displayed")
        if node_errors:
            warnings.append(f"{len(node_errors)} concept fetch errors")
        if edge_errors:
            warnings.append(f"{len(edge_errors)} connection fetch errors")

        result = {
            "session_id": self.current_session.id,
            "topic": self.current_session.topic,
            "nodes": nodes,
            "edges": edges,
            "relationship_descriptions": relationship_descriptions,
            "node_count": len(nodes),
            "connection_count": len(edges),
            "success": True
        }

        # Add warnings if there were issues
        if warnings:
            result["warnings"] = warnings
            result["data_issues"] = {
                "missing_nodes": missing_nodes,
                "missing_edges": missing_edges,
                "node_errors": node_errors,
                "edge_errors": edge_errors
            }

        return result

    async def _enqueue_pd_consolidate(self) -> None:
        """Enqueue a PD consolidation task after an artifact or at session end."""
        try:
            if self._pd_queue is not None and self.current_session:
                await self._pd_queue.put({
                    "type": "consolidate",
                    "session_id": self.current_session.id,
                    # Use last 10 messages for consolidation context
                    "context": self.current_session.messages[-10:]
                })
        except Exception as e:
            logger.error(f"Failed to enqueue PD consolidate task: {e}")

    async def _pd_worker_loop(self) -> None:
        """Background worker that processes PD tasks without blocking CA."""
        try:
            while True:
                task = await self._pd_queue.get()
                ttype = task.get("type")
                session_id = task.get("session_id")

                # Ensure session hasn't been switched; allow consolidation even if ended, using IDs in DB
                try:
                    if ttype == "exchange":
                        await self._run_pd_for_exchange(
                            session_id,
                            task.get("user_input", ""),
                            task.get("assistant", ""),
                            task.get("context", [])
                        )
                    elif ttype == "consolidate":
                        await self._run_pd_consolidation(session_id, task.get("context", []))
                except Exception as e:
                    logger.error(f"PD worker task failed: {e}")
                finally:
                    self._pd_queue.task_done()
        except asyncio.CancelledError:
            logger.info("PD worker loop cancelled")

    async def _run_pd_for_exchange(self, session_id: str, user_input: str, assistant: str, context_msgs: List[Dict[str, Any]]) -> None:
        """Run PD for a single exchange and update the graph asynchronously."""
        try:
            extractions = await self.pd.extract_patterns(user_input, assistant, context_msgs)
        except Exception as e:
            logger.error(f"Pattern extraction failed in background: {e}")
            return

        try:
            self._apply_extractions_to_graph(session_id, extractions)
        except Exception as e:
            logger.error(f"Failed to apply PD extractions to graph: {e}")

    async def _run_pd_consolidation(self, session_id: str, context_msgs: List[Dict[str, Any]]) -> None:
        """Run a consolidation PD pass using recent context to ensure graph consistency."""
        try:
            # Build a minimal synthetic input by concatenating recent exchanges
            user_concat = " ".join([m.get("user", "") for m in context_msgs])
            assistant_concat = " ".join(
                [m.get("assistant", "") for m in context_msgs])
            extractions = await self.pd.extract_patterns(user_concat, assistant_concat, context_msgs)
            self._apply_extractions_to_graph(session_id, extractions)
        except Exception as e:
            logger.error(f"PD consolidation failed: {e}")

    def _apply_extractions_to_graph(self, session_id: str, extractions: List[Any]) -> Tuple[int, int]:
        """Create nodes and edges in the graph from PD extractions. Returns (#nodes, #edges)."""
        new_nodes = 0
        new_edges = 0

        for extraction in extractions or []:
            # Create concept node
            concept_node = Node(
                id=str(uuid.uuid4()),
                label=getattr(extraction, "concept", ""),
                node_type=NodeType.CONCEPT,
                properties={
                    "domain": getattr(extraction, "domain", "unknown"),
                    "context": getattr(extraction, "context", ""),
                    "confidence": getattr(extraction, "confidence", 0.5),
                    "session_id": session_id,
                },
            )

            if self.graph_db.create_node(concept_node):
                new_nodes += 1
                # Attach to in-memory session if still the same
                if self.current_session and self.current_session.id == session_id:
                    self.current_session.nodes_created.append(concept_node.id)

            # Metaphor nodes and edges
            for metaphor in getattr(extraction, "metaphors", []) or []:
                metaphor_node = Node(
                    id=str(uuid.uuid4()),
                    label=metaphor,
                    node_type=NodeType.METAPHOR,
                    properties={
                        "concept": getattr(extraction, "concept", ""),
                        "session_id": session_id,
                    },
                )
                if self.graph_db.create_node(metaphor_node):
                    new_nodes += 1
                    edge = Edge(
                        id=str(uuid.uuid4()),
                        source_id=concept_node.id,
                        target_id=metaphor_node.id,
                        edge_type=EdgeType.METAPHORICAL,
                        properties={"session_id": session_id},
                    )
                    if self.graph_db.create_edge(edge):
                        new_edges += 1
                        if self.current_session and self.current_session.id == session_id:
                            self.current_session.edges_created.append(edge.id)

        return new_nodes, new_edges

    async def _command_analyze(self, apply: bool = False) -> Dict[str, Any]:
        """Run MA analysis on demand. Optionally apply persona adjustments."""
        if not self.current_session:
            return {"error": "No active session"}

        # Build a minimal set of recent extractions for context by reusing PD quickly (non-blocking idea)
        recent_context = self.current_session.messages[-5:]
        # Do not block on PD; pass empty extractions for analysis context by default
        extractions: List[Any] = []

        try:
            analysis = await self.ma.analyze_session(self.current_session, extractions, {})
            self._last_ma_analysis = analysis
        except Exception as e:
            logger.error(f"MA analyze command failed: {e}")
            return {"error": f"MA analyze failed: {e}"}

        applied = False
        if apply and isinstance(analysis, dict):
            adjustments = analysis.get("persona_adjustments") or {}
            if isinstance(adjustments, dict) and adjustments:
                try:
                    await self.ca.update_persona(adjustments)
                    applied = True
                except Exception as e:
                    logger.error(f"Applying MA adjustments failed: {e}")

        return {"analysis": analysis, "applied": applied}

    async def _command_apply_ma(self) -> Dict[str, Any]:
        """Apply the last MA persona adjustments if available."""
        if not self.current_session:
            return {"error": "No active session"}
        if not self._last_ma_analysis:
            return {"error": "No MA analysis available. Run /analyze first."}
        adjustments = self._last_ma_analysis.get("persona_adjustments") if isinstance(
            self._last_ma_analysis, dict) else None
        if not (isinstance(adjustments, dict) and adjustments):
            return {"message": "No persona adjustments to apply."}
        try:
            await self.ca.update_persona(adjustments)
            return {"message": "Persona adjustments applied.", "applied": True, "adjustments": adjustments}
        except Exception as e:
            logger.error(f"Failed to apply MA adjustments: {e}")
            return {"error": f"Failed to apply MA adjustments: {e}"}

    def _create_fallback_relationship_descriptions(self, nodes: List[Dict], edges: List[Dict]) -> List[str]:
        """Create simple fallback relationship descriptions when LLM generation fails."""
        descriptions = []

        # Create a lookup for node labels
        node_labels = {node.get('id'): node.get(
            'label', 'Unknown') for node in nodes}

        for edge in edges:
            try:
                source_id = edge.get('source_id')
                target_id = edge.get('target_id')
                edge_type = edge.get('edge_type', 'relates')

                source_label = node_labels.get(
                    source_id, edge.get('source_label', 'Unknown'))
                target_label = node_labels.get(
                    target_id, edge.get('target_label', 'Unknown'))

                # Create simple description
                if edge_type == 'metaphorical':
                    desc = f"• {source_label} is understood through the metaphor of {target_label}"
                elif edge_type == 'conceptual':
                    desc = f"• {source_label} connects conceptually to {target_label}"
                elif edge_type == 'causal':
                    desc = f"• {source_label} influences or causes {target_label}"
                else:
                    desc = f"• {source_label} relates to {target_label} through {edge_type}"

                descriptions.append(desc)

            except Exception as e:
                logger.warning(
                    f"Error creating fallback description for edge: {e}")
                descriptions.append("• Connection details unavailable")

        return descriptions

    async def _generate_relationship_descriptions(self, nodes: List[Dict], edges: List[Dict]) -> List[str]:
        """Generate natural language descriptions of concept relationships using LLM."""
        if not edges:
            return []

        # Build context about the session and concepts
        session_context = f"Session Topic: {self.current_session.topic}\n"

        # Create concept summaries
        concept_summaries = {}
        for node in nodes:
            label = node.get('label', 'Unknown')
            node_type = node.get('node_type', 'concept')
            domain = node.get('properties', {}).get('domain', 'general')
            context = node.get('properties', {}).get('context', '')

            concept_summaries[label] = {
                'type': node_type,
                'domain': domain,
                'context': context[:100] + '...' if len(context) > 100 else context
            }

        # Build relationship data for LLM
        relationships_data = []
        for edge in edges:
            source_label = edge.get('source_label', 'Unknown')
            target_label = edge.get('target_label', 'Unknown')
            edge_type = edge.get('edge_type', 'relates')

            relationships_data.append({
                'source': source_label,
                'target': target_label,
                'type': edge_type,
                'source_info': concept_summaries.get(source_label, {}),
                'target_info': concept_summaries.get(target_label, {})
            })

        # Get recent conversation context for relationship understanding
        recent_context = ""
        if self.current_session.messages:
            # Last 4 exchanges
            recent_messages = self.current_session.messages[-4:]
            recent_context = "\n".join([
                f"User: {msg.get('user', '')}\nAssistant: {msg.get('assistant', '')}"
                for msg in recent_messages
            ])

        prompt = f"""
You are describing the conceptual relationships discovered in a learning session. Create natural, insightful descriptions that help the learner understand how their concepts connect.

{session_context}

CONCEPTS EXPLORED:
{json.dumps(concept_summaries, indent=2)}

RELATIONSHIPS TO DESCRIBE:
{json.dumps(relationships_data, indent=2)}

RECENT CONVERSATION CONTEXT:
{recent_context}

For each relationship, create a natural language description that:
1. Explains how the concepts connect in the context of this exploration
2. Uses language that reflects the learner's metaphorical thinking
3. Highlights the significance of the connection
4. Is engaging and insightful, not just technical

Format each description as a complete sentence starting with "•". Keep descriptions concise but meaningful (1-2 sentences each).

Example format:
• The concept of "quantum superposition" emerges from your exploration of "uncertainty," showing how your intuition about "multiple possibilities existing simultaneously" bridges abstract physics with everyday decision-making.
• Your metaphor of "energy flow" creates a powerful connection between "thermodynamics" and "biological systems," revealing how the same principles govern both engines and living cells.
"""

        try:
            # Add timeout for performance optimization
            response = await asyncio.wait_for(
                self.ca.client.chat.completions.create(
                    model=self.ca._get_deployment_name(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,  # Creative but focused
                    max_tokens=400
                ),
                timeout=20.0  # 20 second timeout for relationship descriptions
            )

            descriptions_text = response.choices[0].message.content.strip()

            # Extract bullet points
            descriptions = []
            for line in descriptions_text.split('\n'):
                line = line.strip()
                if line.startswith('•'):
                    descriptions.append(line)

            logger.info(
                f"Generated {len(descriptions)} relationship descriptions")
            return descriptions

        except asyncio.TimeoutError:
            logger.warning(
                "Relationship description generation timed out, using fallback")
            return self._create_fallback_relationship_descriptions(nodes, edges)

        except Exception as e:
            logger.error(f"Failed to generate relationship descriptions: {e}")
            # Fallback to simple descriptions
            return self._create_fallback_relationship_descriptions(nodes, edges)

    async def _gap_check(self) -> Dict[str, Any]:
        """Check gaps between user understanding and canonical knowledge with comprehensive error handling."""
        try:
            # Validate session state
            if not self.current_session:
                return {
                    "error": "No active session found",
                    "message": "🔄 **Start a session first**\n\nUse a topic to begin exploring, then try gap check again.\n\n*Example: Start discussing quantum mechanics, then use `/gapcheck` to analyze your understanding.*",
                    "recovery_suggestions": ["Start a new session with a topic", "Engage in conversation about concepts", "Try gap check after discussing specific ideas"]
                }

            if not self.current_session.messages:
                return {
                    "error": "No conversation to analyze",
                    "message": "💭 **Let's explore some ideas first**\n\nGap check works best after you've discussed concepts and shared your thoughts.\n\n*Try explaining a concept or asking questions, then use `/gapcheck` to see how your understanding compares to canonical knowledge.*",
                    "recovery_suggestions": ["Share your thoughts on the session topic", "Ask questions about concepts", "Explain ideas using your own words and metaphors"]
                }

            # 1. Identify most recent concept with error handling
            recent_concept = None
            try:
                recent_concept = await self._identify_recent_concept()
            except Exception as e:
                logger.error(f"Concept identification failed: {e}")
                return {
                    "error": "Failed to identify concept from conversation",
                    "message": "🤔 **Having trouble identifying the main concept**\n\nThis might happen if the conversation covers many topics. Try discussing a specific concept more directly.\n\n*Example: 'I think of quantum superposition like...' or 'My understanding of entropy is...'*",
                    "recovery_suggestions": ["Focus on one specific concept", "Use clear concept names in your explanations", "Try rephrasing your thoughts about the topic"]
                }

            if not recent_concept:
                return {
                    "error": "No clear concept identified from recent conversation",
                    "message": "🎯 **Let's focus on a specific concept**\n\nGap check works best when we can identify a clear concept from your recent discussion.\n\n*Try mentioning specific terms or explaining particular ideas you're exploring.*",
                    "recovery_suggestions": ["Mention specific concept names", "Explain particular ideas in detail", "Ask about specific aspects of the topic"]
                }

            # 2. Retrieve u_vector with error handling
            u_vector = None
            try:
                u_vector = await self._get_user_vector(recent_concept)
            except Exception as e:
                logger.error(
                    f"Failed to retrieve u_vector for {recent_concept}: {e}")
                return {
                    "error": f"Database error retrieving understanding for '{recent_concept}'",
                    "message": f"⚠️ **Technical issue accessing your understanding of '{recent_concept}'**\n\nThis might be a temporary database issue. Your exploration data is safe.\n\n*Try the gap check again in a moment, or continue exploring other concepts.*",
                    "recovery_suggestions": ["Try gap check again in a moment", "Continue exploring other concepts", "Check if the database connection is stable"]
                }

            if not u_vector:
                return {
                    "error": f"No user understanding vector found for '{recent_concept}'",
                    "message": f"📚 **Need more exploration of '{recent_concept}'**\n\nI haven't captured enough of your personal understanding yet. Share more of your thoughts, metaphors, or explanations about this concept.\n\n*Try explaining how you think about '{recent_concept}' or what it reminds you of.*",
                    "recovery_suggestions": [f"Explain your understanding of '{recent_concept}' in your own words", f"Share metaphors or analogies for '{recent_concept}'", f"Discuss what '{recent_concept}' means to you"]
                }

            # 3. Get or create c_vector with comprehensive error handling
            c_vector = None
            try:
                c_vector = await self._get_or_create_canonical_vector(recent_concept)
            except Exception as e:
                logger.error(
                    f"Failed to get/create c_vector for {recent_concept}: {e}")
                # Check if it's an API issue or database issue
                if "openai" in str(e).lower() or "api" in str(e).lower():
                    return {
                        "error": f"AI service unavailable for canonical analysis of '{recent_concept}'",
                        "message": f"🌐 **AI service temporarily unavailable**\n\nI can't generate the canonical definition for '{recent_concept}' right now due to connectivity issues.\n\n*Your exploration continues! Try gap check again later, or explore other aspects of the topic.*",
                        "recovery_suggestions": ["Try gap check again later", "Continue exploring other concepts", "Check internet connectivity"]
                    }
                else:
                    return {
                        "error": f"Failed to obtain canonical vector for '{recent_concept}'",
                        "message": f"⚠️ **Technical issue with canonical knowledge for '{recent_concept}'**\n\nThere was a problem accessing or creating the canonical understanding. This doesn't affect your exploration.\n\n*Continue exploring, and we can try gap analysis again later.*",
                        "recovery_suggestions": ["Continue exploring the concept", "Try gap check again later", "Explore related concepts"]
                    }

            if not c_vector:
                return {
                    "error": f"Failed to obtain canonical vector for '{recent_concept}'",
                    "message": f"📖 **Canonical knowledge unavailable for '{recent_concept}'**\n\nI couldn't access or generate the standard academic understanding of this concept right now.\n\n*Your personal exploration is still valuable! Continue developing your understanding.*",
                    "recovery_suggestions": ["Continue exploring your understanding", "Try gap check with other concepts", "Explore related ideas"]
                }

            # 4. Calculate gap with error handling
            gap_analysis = None
            try:
                gap_analysis = self.embedding_service.calculate_gap_score(
                    u_vector, c_vector)
            except Exception as e:
                logger.error(
                    f"Gap calculation failed for {recent_concept}: {e}")
                return {
                    "error": f"Failed to calculate understanding gap for '{recent_concept}'",
                    "message": f"🔢 **Analysis calculation issue**\n\nThere was a problem comparing your understanding with canonical knowledge for '{recent_concept}'.\n\n*This might be due to vector dimension mismatches or calculation errors. Your exploration data is intact.*",
                    "recovery_suggestions": ["Try gap check again", "Continue exploring the concept", "Try gap check with other concepts"]
                }

            # 5. Generate user message with fallback
            message = None
            try:
                message = await self._format_gap_message(recent_concept, gap_analysis)
            except Exception as e:
                logger.error(f"Failed to format gap message: {e}")
                message = self._fallback_gap_message(
                    recent_concept, gap_analysis)

            return {
                "concept": recent_concept,
                "similarity": gap_analysis["similarity"],
                "gap_score": gap_analysis["gap_score"],
                "severity": gap_analysis["severity"],
                "message": message,
                "success": True
            }

        except Exception as e:
            logger.error(f"Unexpected error in gap check: {e}")
            return {
                "error": "Unexpected gap check failure",
                "message": "⚠️ **Unexpected issue with gap analysis**\n\nSomething unexpected happened during the gap check process. Your exploration data is safe.\n\n*Try continuing your exploration and attempt gap check again later.*",
                "recovery_suggestions": ["Continue your exploration", "Try gap check again later", "Restart the session if issues persist"]
            }

    async def _inject_seed(self) -> Dict[str, Any]:
        """Inject a new exploration seed from MA."""
        if not self.current_session:
            return {"error": "No active session"}

        seed = await self.ma.generate_seed(self.current_session)
        return {
            "seed_concept": seed.concept,
            "rationale": seed.rationale,
            "suggested_questions": seed.suggested_questions
        }

    async def _identify_recent_concept(self) -> Optional[str]:
        """Extract the most recently discussed concept using a direct LLM call with conversation history limiting for performance."""
        if not self.current_session or not self.current_session.messages:
            return None

        # Performance optimization: limit to last 4 exchanges (8 messages) for faster processing
        # Last 4 exchanges only
        recent_messages = self.current_session.messages[-4:]

        # Further optimize by truncating very long messages
        recent_messages_formatted = []
        for msg in recent_messages:
            user_msg = msg.get('user', '')
            assistant_msg = msg.get('assistant', '')

            # Truncate messages if they're too long (performance optimization)
            if len(user_msg) > 500:
                user_msg = user_msg[:500] + "..."
            if len(assistant_msg) > 500:
                assistant_msg = assistant_msg[:500] + "..."

            recent_messages_formatted.append(
                f"User: {user_msg}\nAssistant: {assistant_msg}")

        conversation_text = "\n".join(recent_messages_formatted)

        prompt = f"""
Given the following recent conversation history, what is the single, most prominent concept being discussed?
Respond with only the concept name and nothing else.

Conversation:
---
{conversation_text}
---

Concept:"""

        try:
            # Add timeout for performance
            response = await asyncio.wait_for(
                self.ca.client.chat.completions.create(
                    model=self.ca._get_deployment_name(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=50
                ),
                timeout=15.0  # 15 second timeout for concept identification
            )
            concept = response.choices[0].message.content.strip().replace(
                '"', '')
            logger.info(f"Identified recent concept via LLM: {concept}")
            return concept
        except asyncio.TimeoutError:
            logger.warning(
                "Concept identification timed out, using session topic as fallback")
            return self.current_session.topic
        except Exception as e:
            logger.error(f"LLM-based concept identification failed: {e}")
            # Fallback to topic as a last resort
            return self.current_session.topic

    async def _get_user_vector(self, concept: str) -> Optional[Vector]:
        """Retrieve u_vector for a concept from Neo4j."""
        try:
            # Search for nodes with this concept label in current session
            session_nodes = []
            for node_id in self.current_session.nodes_created:
                node_data = self.graph_db.get_node(node_id)
                if node_data and node_data.get('label', '').lower() == concept.lower():
                    session_nodes.append(node_data)

            # Find the most recent concept node with a u_vector
            for node_data in reversed(session_nodes):  # Most recent first
                if node_data.get('u_vector'):
                    u_vector_data = node_data['u_vector']
                    return Vector(
                        values=u_vector_data.get('values', []),
                        model=u_vector_data.get(
                            'model', 'text-embedding-ada-002'),
                        dimension=u_vector_data.get('dimension', 1536)
                    )

            # If no u_vector in current session, search globally
            all_concept_nodes = self.graph_db.search_nodes_by_content(
                concept, limit=10)
            for node_data in all_concept_nodes:
                if (node_data.get('node_type') == 'concept' and
                    node_data.get('label', '').lower() == concept.lower() and
                        node_data.get('u_vector')):
                    u_vector_data = node_data['u_vector']
                    return Vector(
                        values=u_vector_data.get('values', []),
                        model=u_vector_data.get(
                            'model', 'text-embedding-ada-002'),
                        dimension=u_vector_data.get('dimension', 1536)
                    )

        except Exception as e:
            logger.error(f"Failed to retrieve u_vector for {concept}: {e}")

        return None

    async def _get_or_create_canonical_vector(self, concept: str) -> Optional[Vector]:
        """Fetch existing c_vector or generate new one with graceful degradation when OpenAI API is unavailable."""
        try:
            # First, try to find existing c_vector in the database
            all_concept_nodes = []
            try:
                all_concept_nodes = self.graph_db.search_nodes_by_content(
                    concept, limit=10)
            except Exception as db_error:
                logger.error(
                    f"Database search failed for {concept}: {db_error}")
                # Continue with empty list - we'll try to create new vector

            # Check existing nodes for c_vector
            for node_data in all_concept_nodes:
                try:
                    if (node_data.get('node_type') == 'concept' and
                        node_data.get('label', '').lower() == concept.lower() and
                            node_data.get('c_vector')):
                        c_vector_data = node_data['c_vector']
                        logger.info(f"Found existing c_vector for {concept}")
                        return Vector(
                            values=c_vector_data.get('values', []),
                            model=c_vector_data.get(
                                'model', 'text-embedding-ada-002'),
                            dimension=c_vector_data.get('dimension', 1536),
                            metadata=c_vector_data.get('metadata', {})
                        )
                except Exception as node_error:
                    logger.warning(
                        f"Error processing existing node for {concept}: {node_error}")
                    continue

            # If no existing c_vector, try to generate a new one with graceful degradation
            logger.info(
                f"No existing c_vector found for {concept}, attempting to create new one")

            # Step 1: Detect domain with fallback
            domain = "general"  # Default fallback
            try:
                domain = await self._detect_concept_domain(concept)
            except Exception as domain_error:
                logger.warning(
                    f"Domain detection failed for {concept}, using 'general': {domain_error}")
                # Use session topic for basic domain inference as fallback
                if self.current_session and self.current_session.topic:
                    topic_lower = self.current_session.topic.lower()
                    if any(word in topic_lower for word in ['physics', 'quantum', 'energy']):
                        domain = 'physics'
                    elif any(word in topic_lower for word in ['chemistry', 'molecule', 'reaction']):
                        domain = 'chemistry'
                    elif any(word in topic_lower for word in ['biology', 'life', 'evolution']):
                        domain = 'biology'
                    elif any(word in topic_lower for word in ['math', 'equation', 'number']):
                        domain = 'mathematics'
                    elif any(word in topic_lower for word in ['philosophy', 'ethics', 'consciousness']):
                        domain = 'philosophy'

            # Step 2: Try to create c_vector with multiple fallback strategies
            c_vector = None
            canonical_definition = f"Standard academic definition of {concept} in {domain} domain"

            try:
                # Primary attempt: Full LLM-powered generation
                c_vector = await self.embedding_service.create_c_vector(concept, domain)
                if c_vector and c_vector.metadata:
                    canonical_definition = c_vector.metadata.get(
                        "canonical_definition", canonical_definition)
                logger.info(
                    f"Successfully created c_vector for {concept} using LLM")

            except Exception as llm_error:
                logger.warning(
                    f"LLM-powered c_vector creation failed for {concept}: {llm_error}")

                # Fallback 1: Try with basic definition and embedding
                try:
                    basic_definition = f"The concept of {concept} in the context of {domain}. This represents the formal academic understanding of {concept}."
                    embedding_values = await self.embedding_service._get_embedding(basic_definition)

                    if embedding_values and len(embedding_values) > 0:
                        c_vector = Vector(
                            values=embedding_values,
                            model="text-embedding-ada-002",
                            dimension=len(embedding_values),
                            metadata={
                                "canonical_definition": basic_definition,
                                "concept": concept,
                                "domain": domain,
                                "vector_type": "c_vector",
                                "fallback_method": "basic_definition"
                            }
                        )
                        canonical_definition = basic_definition
                        logger.info(
                            f"Created fallback c_vector for {concept} using basic definition")

                except Exception as fallback_error:
                    logger.warning(
                        f"Basic definition fallback failed for {concept}: {fallback_error}")

                    # Fallback 2: Create minimal vector from concept name only
                    try:
                        minimal_text = f"{concept} {domain}"
                        embedding_values = await self.embedding_service._get_embedding(minimal_text)

                        if embedding_values and len(embedding_values) > 0:
                            c_vector = Vector(
                                values=embedding_values,
                                model="text-embedding-ada-002",
                                dimension=len(embedding_values),
                                metadata={
                                    "canonical_definition": f"Minimal representation of {concept}",
                                    "concept": concept,
                                    "domain": domain,
                                    "vector_type": "c_vector",
                                    "fallback_method": "minimal_text"
                                }
                            )
                            canonical_definition = f"Minimal representation of {concept}"
                            logger.info(
                                f"Created minimal c_vector for {concept} from concept name")

                    except Exception as minimal_error:
                        logger.error(
                            f"All c_vector creation methods failed for {concept}: {minimal_error}")
                        return None

            if not c_vector:
                logger.error(f"Failed to create any c_vector for {concept}")
                return None

            # Step 3: Try to save c_vector to Neo4j with error handling
            try:
                concept_node_id = None

                # Try to find existing concept node to update
                for node_data in all_concept_nodes:
                    if (node_data.get('node_type') == 'concept' and
                            node_data.get('label', '').lower() == concept.lower()):
                        concept_node_id = node_data.get('id')
                        break

                # Create or update the node with c_vector
                if concept_node_id:
                    # Update existing node
                    try:
                        node_data = self.graph_db.get_node(concept_node_id)
                        if node_data:
                            updated_node = Node(
                                id=concept_node_id,
                                label=concept,
                                node_type=NodeType.CONCEPT,
                                properties={
                                    **node_data.get('properties', {}),
                                    'domain': domain,
                                    'canonical_definition': canonical_definition
                                },
                                u_vector=Vector(
                                    **node_data['u_vector']) if node_data.get('u_vector') else None,
                                c_vector=c_vector
                            )
                            self.graph_db.create_node(updated_node)
                            logger.info(
                                f"Updated existing node {concept_node_id} with c_vector for {concept}")
                    except Exception as update_error:
                        logger.warning(
                            f"Failed to update existing node for {concept}: {update_error}")
                        # Continue - we still have the c_vector even if saving failed
                else:
                    # Create new node with c_vector
                    try:
                        new_node = Node(
                            id=str(uuid.uuid4()),
                            label=concept,
                            node_type=NodeType.CONCEPT,
                            properties={
                                'domain': domain,
                                'canonical_definition': canonical_definition,
                                'session_id': self.current_session.id if self.current_session else None
                            },
                            c_vector=c_vector
                        )
                        self.graph_db.create_node(new_node)
                        logger.info(
                            f"Created new node with c_vector for {concept}")
                    except Exception as create_error:
                        logger.warning(
                            f"Failed to create new node for {concept}: {create_error}")
                        # Continue - we still have the c_vector even if saving failed

            except Exception as save_error:
                logger.warning(
                    f"Failed to save c_vector to database for {concept}: {save_error}")
                # Continue - we still return the c_vector even if saving failed

            return c_vector

        except Exception as e:
            logger.error(
                f"Unexpected error in _get_or_create_canonical_vector for {concept}: {e}")
            return None

    async def _detect_concept_domain(self, concept: str) -> str:
        """Use LLM to intelligently detect the most appropriate domain for a concept."""

        # Get session context for domain detection
        session_context = ""
        if self.current_session:
            session_context = f"Session topic: {self.current_session.topic}\n"
            if self.current_session.messages:
                recent_messages = self.current_session.messages[-3:]
                session_context += "Recent conversation:\n" + "\n".join([
                    f"User: {msg.get('user', '')}\nAssistant: {msg.get('assistant', '')}"
                    for msg in recent_messages
                ])

        prompt = f"""
Determine the most appropriate academic domain for the concept "{concept}" based on the context.

{session_context}

Consider these domains:
- physics (fundamental forces, matter, energy, quantum mechanics, relativity)
- chemistry (molecules, reactions, bonds, thermodynamics, materials)
- biology (life processes, evolution, genetics, ecology, physiology)
- mathematics (numbers, equations, proofs, logic, geometry, statistics)
- philosophy (ethics, logic, metaphysics, epistemology, consciousness)
- computer_science (algorithms, data structures, programming, AI, systems)
- psychology (cognition, behavior, learning, perception, social dynamics)
- economics (markets, trade, value, incentives, systems)
- engineering (design, systems, optimization, problem-solving)
- general (interdisciplinary or everyday concepts)

Based on the concept "{concept}" and the conversation context, what is the single most appropriate domain?

Respond with just the domain name (e.g., "physics", "biology", "general").
"""

        try:
            response = await self.ca.client.chat.completions.create(
                model=self.ca._get_deployment_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low temperature for consistent domain classification
                max_tokens=50
            )

            domain = response.choices[0].message.content.strip().lower()

            # Validate domain is in our expected list
            valid_domains = [
                'physics', 'chemistry', 'biology', 'mathematics', 'philosophy',
                'computer_science', 'psychology', 'economics', 'engineering', 'general'
            ]

            if domain in valid_domains:
                logger.info(
                    f"Detected domain '{domain}' for concept '{concept}'")
                return domain
            else:
                logger.warning(
                    f"Invalid domain '{domain}' detected, defaulting to 'general'")
                return 'general'

        except Exception as e:
            logger.error(f"Domain detection failed for {concept}: {e}")
            # Fallback: use session topic or default to general
            if self.current_session and self.current_session.topic:
                # Simple keyword matching as fallback
                topic_lower = self.current_session.topic.lower()
                if any(word in topic_lower for word in ['physics', 'quantum', 'energy', 'force']):
                    return 'physics'
                elif any(word in topic_lower for word in ['chemistry', 'molecule', 'reaction', 'chemical']):
                    return 'chemistry'
                elif any(word in topic_lower for word in ['biology', 'life', 'evolution', 'genetic']):
                    return 'biology'
                elif any(word in topic_lower for word in ['math', 'equation', 'number', 'calculate']):
                    return 'mathematics'
                elif any(word in topic_lower for word in ['philosophy', 'ethics', 'consciousness', 'meaning']):
                    return 'philosophy'
                elif any(word in topic_lower for word in ['computer', 'algorithm', 'programming', 'ai']):
                    return 'computer_science'
                elif any(word in topic_lower for word in ['psychology', 'behavior', 'mind', 'cognitive']):
                    return 'psychology'
                elif any(word in topic_lower for word in ['economics', 'market', 'trade', 'money']):
                    return 'economics'
                elif any(word in topic_lower for word in ['engineering', 'design', 'system', 'build']):
                    return 'engineering'

            return 'general'

    async def _format_gap_message(self, concept: str, gap_analysis: Dict[str, Any]) -> str:
        """Create personalized, contextual gap analysis message using LLM."""
        similarity = gap_analysis.get('similarity', 0.0)
        gap_score = gap_analysis.get('gap_score', 1.0)
        severity = gap_analysis.get('severity', 'unknown')
        canonical_definition = gap_analysis.get('canonical_definition', '')

        # Get recent conversation context for personalization
        recent_context = ""
        if self.current_session and self.current_session.messages:
            # Last 3 exchanges
            recent_messages = self.current_session.messages[-3:]
            recent_context = "\n".join([
                f"User: {msg.get('user', '')}\nAssistant: {msg.get('assistant', '')}"
                for msg in recent_messages
            ])

        # Extract user's metaphors and explanations from recent context
        user_metaphors = await self._extract_user_metaphors(concept, recent_context)

        prompt = f"""
You are providing personalized feedback on a learner's understanding gap analysis. Be encouraging, insightful, and specific to their learning journey.

CONCEPT: {concept}
SIMILARITY SCORE: {similarity:.2f} ({int(similarity * 100)}% alignment)
GAP SCORE: {gap_score:.2f}
SEVERITY: {severity}

CANONICAL DEFINITION:
{canonical_definition}

USER'S RECENT EXPLORATION:
{recent_context}

USER'S METAPHORS DETECTED:
{user_metaphors}

SESSION TOPIC: {self.current_session.topic if self.current_session else 'General exploration'}

Create a personalized gap analysis message that:
1. Acknowledges their specific metaphors and thinking patterns
2. Explains what the gap means in the context of their exploration
3. Highlights what they're doing well
4. Suggests specific ways to bridge the gap using their preferred learning style
5. Maintains an encouraging, curious tone that fits the Socratic learning approach
6. Uses appropriate emoji and formatting for engagement

Keep it conversational and specific to their journey, not generic. Reference their actual metaphors and thinking patterns.
"""

        try:
            response = await self.ca.client.chat.completions.create(
                model=self.ca._get_deployment_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Creative but focused
                max_tokens=400
            )

            personalized_message = response.choices[0].message.content.strip()
            logger.info(
                f"Generated personalized gap analysis message for {concept}")
            return personalized_message

        except Exception as e:
            logger.error(f"Failed to generate personalized gap message: {e}")
            # Fallback to simpler programmatic version
            return self._fallback_gap_message(concept, gap_analysis)

    def _fallback_gap_message(self, concept: str, gap_analysis: Dict[str, Any]) -> str:
        """Fallback gap message if LLM generation fails."""
        similarity = gap_analysis.get('similarity', 0.0)
        severity = gap_analysis.get('severity', 'unknown')
        similarity_percent = int(similarity * 100)

        if severity in ['minimal', 'low']:
            return f"🎯 **Gap Analysis for '{concept}'**\n\nYour understanding shows **{similarity_percent}% alignment** with canonical knowledge - excellent work! You've grasped the core concepts well."
        elif severity == 'moderate':
            return f"🔍 **Gap Analysis for '{concept}'**\n\nYour understanding shows **{similarity_percent}% alignment** with canonical knowledge. There's room to deepen your grasp - a great learning opportunity!"
        else:
            return f"🚀 **Gap Analysis for '{concept}'**\n\nYour understanding shows **{similarity_percent}% alignment** with canonical knowledge. You're approaching this from a unique angle - let's explore the canonical perspective!"

    async def _extract_user_metaphors(self, concept: str, recent_context: str) -> str:
        """Extract user's metaphors and thinking patterns for personalization."""
        if not recent_context:
            return "No recent metaphors detected"

        prompt = f"""
Analyze this conversation for metaphors, analogies, and thinking patterns the user employed when discussing "{concept}":

{recent_context}

Extract:
1. Specific metaphors used (e.g., "like water flowing", "building blocks", "dance")
2. Analogies drawn to other domains
3. Thinking patterns (visual, mechanical, organic, mathematical, etc.)

Return a concise summary of their metaphorical approach.
"""

        try:
            response = await self.ca.client.chat.completions.create(
                model=self.ca._get_deployment_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to extract user metaphors: {e}")
            return "Unable to analyze metaphorical patterns"
