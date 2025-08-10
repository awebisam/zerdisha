"""Core orchestrator that manages CA, PD, and MA agents."""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from ..models.graph import Session, Node, Edge, NodeType, EdgeType
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
        # This would analyze u-vectors vs c-vectors
        return {"message": "Gap check analysis (to be implemented)"}
    
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