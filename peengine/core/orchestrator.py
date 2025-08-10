"""Core orchestrator that manages CA, PD, and MA agents."""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
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
    """Main orchestrator for the Personal Exploration Engine."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Initialize databases
        self.graph_db = Neo4jClient(settings.database_config)
        self.message_db = MongoDBClient(settings.mongodb_uri, settings.mongodb_database)
        
        # Initialize agents
        self.ca = ConversationalAgent(settings.llm_config, settings.persona_config)
        self.pd = PatternDetector(settings.llm_config)
        self.ma = MetacognitiveAgent(settings.llm_config)
        
        # Initialize services
        self.embedding_service = EmbeddingService(settings.llm_config)
        self.analytics = AnalyticsEngine(self.graph_db, self.embedding_service)
        
        # Current session
        self.current_session: Optional[Session] = None
        
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
        
        logger.info("Exploration Engine initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the engine cleanly."""
        if self.current_session and self.current_session.status == "active":
            await self.end_session()
        
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
        
        neo4j_success = self.graph_db.create_session_summary(self.current_session)
        
        if not (mongo_success and neo4j_success):
            raise RuntimeError(f"Failed to create session in databases")
        
        # Load relevant past nodes for context
        relevant_nodes = await self._load_relevant_context(topic)
        
        # Initialize CA with topic and context
        await self.ca.start_session(topic, relevant_nodes)
        
        logger.info(f"Started session: {session_id} - {session_title}")
        return self.current_session
    
    async def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input through the CA/PD/MA pipeline."""
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
        
        # Extract patterns with Pattern Detector
        try:
            extractions = await self.pd.extract_patterns(
                user_input, 
                ca_response["message"],
                self.current_session.messages[-5:]  # Last 5 exchanges for context
            )
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            extractions = []  # Continue with empty extractions
        
        # Create nodes and edges from extractions
        new_nodes = []
        new_edges = []
        
        for extraction in extractions:
            # Create concept nodes
            concept_node = Node(
                id=str(uuid.uuid4()),
                label=extraction.concept,
                node_type=NodeType.CONCEPT,
                properties={
                    "domain": extraction.domain,
                    "context": extraction.context,
                    "confidence": extraction.confidence,
                    "session_id": self.current_session.id
                }
            )
            
            if self.graph_db.create_node(concept_node):
                new_nodes.append(concept_node.id)
                self.current_session.nodes_created.append(concept_node.id)
            
            # Create metaphor nodes and connections
            for metaphor in extraction.metaphors:
                metaphor_node = Node(
                    id=str(uuid.uuid4()),
                    label=metaphor,
                    node_type=NodeType.METAPHOR,
                    properties={
                        "concept": extraction.concept,
                        "session_id": self.current_session.id
                    }
                )
                
                if self.graph_db.create_node(metaphor_node):
                    new_nodes.append(metaphor_node.id)
                    
                    # Create edge between concept and metaphor
                    edge = Edge(
                        id=str(uuid.uuid4()),
                        source_id=concept_node.id,
                        target_id=metaphor_node.id,
                        edge_type=EdgeType.METAPHORICAL,
                        properties={"session_id": self.current_session.id}
                    )
                    
                    if self.graph_db.create_edge(edge):
                        new_edges.append(edge.id)
                        self.current_session.edges_created.append(edge.id)
        
        # Run metacognitive analysis
        ma_analysis = await self.ma.analyze_session(
            self.current_session,
            extractions,
            ca_response.get("reasoning", {})
        )
        
        # Apply MA persona adjustments immediately if present
        if ma_analysis.get("persona_adjustments"):
            adjustments = ma_analysis["persona_adjustments"]
            logger.info(f"MA provided persona adjustments: {list(adjustments.keys())}")
            await self.ca.update_persona(adjustments)
            logger.info(f"Applied persona adjustments to CA for session {self.current_session.id}")
        
        # Update MongoDB with analysis data
        analysis_data = {
            "ma_insights": ma_analysis.get("insights", []),
            "concepts_extracted": len(new_nodes),
            "ma_flags": ma_analysis.get("flags", []),
            "total_exchanges": len(self.current_session.messages)
        }
        await self.message_db.update_session_analysis(self.current_session.id, analysis_data)
        
        return {
            "message": ca_response["message"],
            "session_id": self.current_session.id,
            "new_concepts": len(new_nodes),
            "ma_insights": ma_analysis.get("insights", []),
            "suggested_commands": ma_analysis.get("suggested_commands", [])
        }
    
    async def execute_command(self, command: str, args: List[str] = None) -> Dict[str, Any]:
        """Execute TUI commands like /map, /gapcheck, /seed."""
        args = args or []
        
        if command == "map":
            return await self._show_session_map()
        elif command == "gapcheck":
            return await self._gap_check()
        elif command == "seed":
            return await self._inject_seed()
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
        
        # Final session analysis
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
        """Show current session's concept map with nodes and contextual relationship descriptions."""
        if not self.current_session:
            return {"error": "No active session"}
        
        # Fetch all nodes created in this session
        nodes = []
        for node_id in self.current_session.nodes_created:
            node_data = self.graph_db.get_node(node_id)
            if node_data:
                nodes.append(node_data)
        
        # Fetch all edges created in this session
        edges = []
        for edge_id in self.current_session.edges_created:
            edge_data = self.graph_db.get_edge(edge_id)
            if edge_data:
                edges.append(edge_data)
        
        # Generate contextual relationship descriptions using LLM
        relationship_descriptions = []
        if edges:
            relationship_descriptions = await self._generate_relationship_descriptions(nodes, edges)
        
        return {
            "session_id": self.current_session.id,
            "topic": self.current_session.topic,
            "nodes": nodes,
            "edges": edges,
            "relationship_descriptions": relationship_descriptions,
            "node_count": len(nodes),
            "connection_count": len(edges)
        }
    
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
            recent_messages = self.current_session.messages[-4:]  # Last 4 exchanges
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
            response = await self.ca.client.chat.completions.create(
                model=self.ca._get_deployment_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,  # Creative but focused
                max_tokens=400
            )
            
            descriptions_text = response.choices[0].message.content.strip()
            
            # Extract bullet points
            descriptions = []
            for line in descriptions_text.split('\n'):
                line = line.strip()
                if line.startswith('•'):
                    descriptions.append(line)
            
            logger.info(f"Generated {len(descriptions)} relationship descriptions")
            return descriptions
            
        except Exception as e:
            logger.error(f"Failed to generate relationship descriptions: {e}")
            # Fallback to simple descriptions
            return [
                f"• {edge.get('source_label', 'Unknown')} connects to {edge.get('target_label', 'Unknown')} through {edge.get('edge_type', 'relationship')}"
                for edge in edges
            ]
    
    async def _gap_check(self) -> Dict[str, Any]:
        """Check gaps between user understanding and canonical knowledge."""
        if not self.current_session or not self.current_session.messages:
            return {"error": "No active session or conversation to analyze"}
        
        # 1. Identify most recent concept
        recent_concept = await self._identify_recent_concept()
        if not recent_concept:
            return {"error": "No clear concept identified from recent conversation"}
        
        # 2. Retrieve u_vector
        u_vector = await self._get_user_vector(recent_concept)
        if not u_vector:
            return {"error": f"No user understanding vector found for '{recent_concept}'"}
        
        # 3. Get or create c_vector
        c_vector = await self._get_or_create_canonical_vector(recent_concept)
        if not c_vector:
            return {"error": f"Failed to obtain canonical vector for '{recent_concept}'"}
        
        # 4. Calculate gap
        gap_analysis = self.embedding_service.calculate_gap_score(u_vector, c_vector)
        
        # 5. Generate user message
        try:
            message = await self._format_gap_message(recent_concept, gap_analysis)
        except Exception as e:
            logger.error(f"Failed to format gap message: {e}")
            message = self._fallback_gap_message(recent_concept, gap_analysis)
        
        return {
            "concept": recent_concept,
            "similarity": gap_analysis["similarity"],
            "gap_score": gap_analysis["gap_score"],
            "severity": gap_analysis["severity"],
            "message": message
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
        """Extract the most recently discussed concept using a direct LLM call for simplicity and accuracy."""
        if not self.current_session or not self.current_session.messages:
            return None

        recent_messages_formatted = "\n".join([
            f"User: {msg.get('user', '')}\nAssistant: {msg.get('assistant', '')}"
            for msg in self.current_session.messages[-6:] # Last 3 exchanges
        ])

        prompt = f"""
Given the following recent conversation history, what is the single, most prominent concept being discussed?
Respond with only the concept name and nothing else.

Conversation:
---
{recent_messages_formatted}
---

Concept:"""

        try:
            response = await self.ca.client.chat.completions.create(
                model=self.ca._get_deployment_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=50
            )
            concept = response.choices[0].message.content.strip().replace('"', '')
            logger.info(f"Identified recent concept via LLM: {concept}")
            return concept
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
                        model=u_vector_data.get('model', 'text-embedding-ada-002'),
                        dimension=u_vector_data.get('dimension', 1536)
                    )
            
            # If no u_vector in current session, search globally
            all_concept_nodes = self.graph_db.search_nodes_by_content(concept, limit=10)
            for node_data in all_concept_nodes:
                if (node_data.get('node_type') == 'concept' and 
                    node_data.get('label', '').lower() == concept.lower() and
                    node_data.get('u_vector')):
                    u_vector_data = node_data['u_vector']
                    return Vector(
                        values=u_vector_data.get('values', []),
                        model=u_vector_data.get('model', 'text-embedding-ada-002'),
                        dimension=u_vector_data.get('dimension', 1536)
                    )
            
        except Exception as e:
            logger.error(f"Failed to retrieve u_vector for {concept}: {e}")
        
        return None
    
    async def _get_or_create_canonical_vector(self, concept: str) -> Optional[Vector]:
        """Fetch existing c_vector or generate new one with enhanced domain intelligence."""
        try:
            # First, try to find existing c_vector in the database
            all_concept_nodes = self.graph_db.search_nodes_by_content(concept, limit=10)
            for node_data in all_concept_nodes:
                if (node_data.get('node_type') == 'concept' and 
                    node_data.get('label', '').lower() == concept.lower() and
                    node_data.get('c_vector')):
                    c_vector_data = node_data['c_vector']
                    return Vector(
                        values=c_vector_data.get('values', []),
                        model=c_vector_data.get('model', 'text-embedding-ada-002'),
                        dimension=c_vector_data.get('dimension', 1536)
                    )
            
            # If no existing c_vector, generate a new one with enhanced domain detection
            domain = await self._detect_concept_domain(concept)
            
            # Generate canonical definition with domain-specific intelligence
            canonical_definition = await self.embedding_service.generate_canonical_definition(concept, domain)
            
            # Create c_vector with enhanced metadata
            c_vector = await self.embedding_service.create_c_vector(concept, domain)
            
            # Save c_vector to Neo4j by updating or creating a concept node
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
                        u_vector=Vector(**node_data['u_vector']) if node_data.get('u_vector') else None,
                        c_vector=c_vector
                    )
                    self.graph_db.create_node(updated_node)
            else:
                # Create new node with c_vector
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
            
            return c_vector
            
        except Exception as e:
            logger.error(f"Failed to get or create c_vector for {concept}: {e}")
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
                logger.info(f"Detected domain '{domain}' for concept '{concept}'")
                return domain
            else:
                logger.warning(f"Invalid domain '{domain}' detected, defaulting to 'general'")
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
            recent_messages = self.current_session.messages[-3:]  # Last 3 exchanges
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
            logger.info(f"Generated personalized gap analysis message for {concept}")
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