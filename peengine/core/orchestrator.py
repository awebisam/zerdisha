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
        extractions = await self.pd.extract_patterns(
            user_input, 
            ca_response["message"],
            self.current_session.messages[-5:]  # Last 5 exchanges for context
        )
        
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
        
        # Apply MA suggestions if any
        if ma_analysis.get("persona_adjustments"):
            await self.ca.update_persona(ma_analysis["persona_adjustments"])
        
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
        """Show current session's concept map."""
        if not self.current_session:
            return {"error": "No active session"}
        
        nodes = []
        for node_id in self.current_session.nodes_created:
            node_data = self.graph_db.get_node(node_id)
            if node_data:
                nodes.append(node_data)
        
        return {
            "session_id": self.current_session.id,
            "topic": self.current_session.topic,
            "nodes": nodes,
            "connections": len(self.current_session.edges_created)
        }
    
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
        message = self._format_gap_message(recent_concept, gap_analysis)
        
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
        """Extract the most recently discussed concept from session messages."""
        if not self.current_session or not self.current_session.messages:
            return None
        
        # Get the last few message exchanges for context
        recent_messages = self.current_session.messages[-3:]  # Last 3 exchanges
        
        # Build conversation text for analysis
        conversation_text = ""
        for msg in recent_messages:
            conversation_text += f"User: {msg.get('user', '')}\n"
            conversation_text += f"Assistant: {msg.get('assistant', '')}\n"
        
        # Use the pattern detector to identify the most prominent concept
        try:
            # Create a simple extraction request focused on the main concept
            extractions = await self.pd.extract_patterns(
                recent_messages[-1].get('user', ''),  # Most recent user input
                recent_messages[-1].get('assistant', ''),  # Most recent assistant response
                recent_messages  # Full context
            )
            
            if extractions:
                # Return the concept with highest confidence from most recent extraction
                best_extraction = max(extractions, key=lambda x: x.confidence)
                return best_extraction.concept
            
        except Exception as e:
            logger.error(f"Failed to extract recent concept: {e}")
        
        # Fallback: try to extract from the session topic or recent messages
        if self.current_session.topic:
            return self.current_session.topic
        
        return None
    
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
        """Fetch existing c_vector or generate new one."""
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
            
            # If no existing c_vector, generate a new one
            # Determine domain from session topic or default to 'general'
            domain = self.current_session.topic if self.current_session else 'general'
            
            # Generate canonical definition
            canonical_definition = await self.embedding_service.generate_canonical_definition(concept, domain)
            
            # Create c_vector
            c_vector = await self.embedding_service.create_c_vector(concept, canonical_definition, domain)
            
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
    
    def _format_gap_message(self, concept: str, gap_analysis: Dict[str, Any]) -> str:
        """Create user-friendly gap analysis message."""
        similarity = gap_analysis.get('similarity', 0.0)
        gap_score = gap_analysis.get('gap_score', 1.0)
        severity = gap_analysis.get('severity', 'unknown')
        
        # Format similarity as percentage
        similarity_percent = int(similarity * 100)
        
        if severity == 'low':
            message = f"🎯 **Gap Analysis for '{concept}'**\n\n"
            message += f"Your understanding shows **{similarity_percent}% alignment** with canonical knowledge - excellent! "
            message += f"You've grasped the core concepts well. The small gap ({gap_score:.2f}) suggests minor "
            message += f"differences in emphasis or perspective that could be worth exploring further."
            
        elif severity == 'medium':
            message = f"🔍 **Gap Analysis for '{concept}'**\n\n"
            message += f"Your understanding shows **{similarity_percent}% alignment** with canonical knowledge. "
            message += f"There's a moderate gap ({gap_score:.2f}) that indicates some key aspects of the canonical "
            message += f"understanding might be missing or emphasized differently in your mental model. "
            message += f"This is a great opportunity to deepen your grasp of the concept."
            
        else:  # high severity
            message = f"🚀 **Gap Analysis for '{concept}'**\n\n"
            message += f"Your understanding shows **{similarity_percent}% alignment** with canonical knowledge. "
            message += f"There's a significant gap ({gap_score:.2f}) which suggests your mental model emphasizes "
            message += f"different aspects than the canonical understanding. This isn't necessarily wrong - "
            message += f"it might indicate you're approaching the concept from a unique angle that could lead to "
            message += f"interesting insights!"
        
        message += f"\n\n*This analysis compares your metaphorical understanding with formal academic definitions.*"
        
        return message