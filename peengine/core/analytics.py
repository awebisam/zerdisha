"""Analytics and review system for sessions and learning patterns."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from ..database.neo4j_client import Neo4jClient
from ..core.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """Analytics engine for session review and learning pattern analysis."""
    
    def __init__(self, db: Neo4jClient, embedding_service: EmbeddingService):
        self.db = db
        self.embedding_service = embedding_service
    
    async def review_session(self, session_id_or_date: str) -> Optional[Dict[str, Any]]:
        """Review a session by ID or date."""
        
        # Try as session ID first
        session_data = self.db.get_session(session_id_or_date)
        
        # If not found, try to search by date
        if not session_data:
            session_data = await self._find_session_by_date(session_id_or_date)
        
        if not session_data:
            return None
        
        # Build comprehensive review
        review = await self._build_session_review(session_data)
        return review
    
    async def _find_session_by_date(self, date_str: str) -> Optional[Dict[str, Any]]:
        """Find session by date string (YYYY-MM-DD)."""
        try:
            target_date = datetime.fromisoformat(date_str)
            start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            
            # Search sessions within the day
            query = """
            MATCH (s:Session)
            WHERE datetime(s.start_time) >= datetime($start) 
            AND datetime(s.start_time) < datetime($end)
            RETURN s
            ORDER BY s.start_time DESC
            LIMIT 1
            """
            
            with self.db.session() as session:
                result = session.run(query, {
                    "start": start_of_day.isoformat(),
                    "end": end_of_day.isoformat()
                })
                record = result.single()
                if record:
                    return dict(record["s"])
            
        except ValueError:
            logger.error(f"Invalid date format: {date_str}")
        except Exception as e:
            logger.error(f"Date search failed: {e}")
        
        return None
    
    async def _build_session_review(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive session review."""
        
        session_id = session_data.get("id")
        
        # Basic session info
        start_time = datetime.fromisoformat(session_data.get("start_time"))
        end_time = None
        if session_data.get("end_time"):
            end_time = datetime.fromisoformat(session_data["end_time"])
        
        duration_minutes = 0
        if end_time:
            duration_minutes = (end_time - start_time).total_seconds() / 60
        
        # Get nodes created in this session
        session_nodes = await self._get_session_nodes(session_id)
        session_edges = await self._get_session_edges(session_id)
        
        # Analyze conversation trajectory
        messages = session_data.get("messages", [])
        conversation_analysis = await self._analyze_conversation_trajectory(messages)
        
        # Analyze concept evolution
        concept_evolution = await self._analyze_concept_evolution(session_nodes, messages)
        
        # Calculate learning metrics
        metrics = await self._calculate_session_metrics(session_data, session_nodes, session_edges)
        
        # Generate insights
        insights = await self._generate_session_insights(
            session_data, session_nodes, session_edges, conversation_analysis
        )
        
        return {
            "session_info": {
                "id": session_id,
                "title": session_data.get("title"),
                "topic": session_data.get("topic"),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat() if end_time else None,
                "duration_minutes": round(duration_minutes, 1),
                "status": session_data.get("status")
            },
            "conversation_summary": {
                "total_exchanges": len(messages),
                "trajectory": conversation_analysis
            },
            "knowledge_graph_changes": {
                "nodes_created": len(session_nodes),
                "edges_created": len(session_edges),
                "concept_evolution": concept_evolution
            },
            "learning_metrics": metrics,
            "insights": insights,
            "session_nodes": session_nodes,
            "session_edges": session_edges
        }
    
    async def _get_session_nodes(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all nodes created in a session."""
        query = """
        MATCH (n:Node)
        WHERE n.properties.session_id = $session_id
        OR $session_id IN n.properties.session_id
        RETURN n
        ORDER BY n.created_at
        """
        
        with self.db.session() as session:
            result = session.run(query, {"session_id": session_id})
            return [dict(record["n"]) for record in result]
    
    async def _get_session_edges(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all edges created in a session."""
        query = """
        MATCH ()-[e:RELATES]->()
        WHERE e.properties.session_id = $session_id
        RETURN e
        ORDER BY e.created_at
        """
        
        with self.db.session() as session:
            result = session.run(query, {"session_id": session_id})
            return [dict(record["e"]) for record in result]
    
    async def _analyze_conversation_trajectory(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the trajectory of the conversation."""
        
        if not messages:
            return {"pattern": "no_conversation"}
        
        # Analyze conversation patterns
        exchange_count = len(messages)
        
        # Look for patterns in message length and complexity
        user_message_lengths = []
        assistant_message_lengths = []
        
        for msg in messages:
            if isinstance(msg, dict):
                user_text = msg.get("user", "")
                assistant_text = msg.get("assistant", "")
                user_message_lengths.append(len(user_text.split()))
                assistant_message_lengths.append(len(assistant_text.split()))
        
        # Calculate trajectory metrics
        if user_message_lengths and assistant_message_lengths:
            avg_user_length = sum(user_message_lengths) / len(user_message_lengths)
            avg_assistant_length = sum(assistant_message_lengths) / len(assistant_message_lengths)
            
            # Analyze progression - are messages getting longer/shorter over time?
            if len(user_message_lengths) > 3:
                early_avg = sum(user_message_lengths[:3]) / 3
                late_avg = sum(user_message_lengths[-3:]) / 3
                progression = "deepening" if late_avg > early_avg * 1.2 else "consistent"
            else:
                progression = "short_session"
        else:
            avg_user_length = 0
            avg_assistant_length = 0
            progression = "unknown"
        
        return {
            "exchange_count": exchange_count,
            "avg_user_message_length": round(avg_user_length, 1),
            "avg_assistant_message_length": round(avg_assistant_length, 1),
            "progression_pattern": progression
        }
    
    async def _analyze_concept_evolution(
        self, 
        session_nodes: List[Dict[str, Any]], 
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze how concepts evolved during the session."""
        
        concepts_by_time = []
        metaphors_by_time = []
        
        for node in session_nodes:
            node_type = node.get("node_type")
            created_at = node.get("created_at")
            
            if node_type == "concept":
                concepts_by_time.append({
                    "concept": node.get("label"),
                    "domain": node.get("properties", {}).get("domain"),
                    "created_at": created_at
                })
            elif node_type == "metaphor":
                metaphors_by_time.append({
                    "metaphor": node.get("label"),
                    "concept": node.get("properties", {}).get("concept"),
                    "created_at": created_at
                })
        
        # Analyze concept introduction pattern
        concept_domains = [c.get("domain") for c in concepts_by_time if c.get("domain")]
        unique_domains = list(set(concept_domains))
        
        return {
            "concepts_introduced": len(concepts_by_time),
            "metaphors_used": len(metaphors_by_time),
            "domains_explored": unique_domains,
            "concept_timeline": concepts_by_time,
            "metaphor_timeline": metaphors_by_time
        }
    
    async def _calculate_session_metrics(
        self, 
        session_data: Dict[str, Any], 
        nodes: List[Dict[str, Any]], 
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate quantitative learning metrics."""
        
        # Basic metrics
        messages = session_data.get("messages", [])
        
        # Concept density (concepts per exchange)
        concept_nodes = [n for n in nodes if n.get("node_type") == "concept"]
        metaphor_nodes = [n for n in nodes if n.get("node_type") == "metaphor"]
        
        exchange_count = len(messages)
        concept_density = len(concept_nodes) / max(exchange_count, 1)
        metaphor_density = len(metaphor_nodes) / max(exchange_count, 1)
        
        # Connection strength (edges per concept)
        connection_strength = len(edges) / max(len(concept_nodes), 1)
        
        # Domain diversity
        domains = set()
        for node in concept_nodes:
            domain = node.get("properties", {}).get("domain")
            if domain:
                domains.add(domain)
        
        domain_diversity = len(domains)
        
        return {
            "concept_density": round(concept_density, 2),
            "metaphor_density": round(metaphor_density, 2), 
            "connection_strength": round(connection_strength, 2),
            "domain_diversity": domain_diversity,
            "total_concepts": len(concept_nodes),
            "total_metaphors": len(metaphor_nodes),
            "total_connections": len(edges)
        }
    
    async def _generate_session_insights(
        self,
        session_data: Dict[str, Any],
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        conversation_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate insights about the session."""
        
        insights = []
        
        # Conversation depth insight
        exchange_count = conversation_analysis.get("exchange_count", 0)
        if exchange_count > 20:
            insights.append({
                "type": "conversation_depth",
                "message": f"Deep exploration with {exchange_count} exchanges - shows sustained engagement",
                "positive": True
            })
        elif exchange_count < 5:
            insights.append({
                "type": "conversation_depth", 
                "message": f"Brief session with {exchange_count} exchanges - consider longer exploration",
                "positive": False
            })
        
        # Concept creation insight
        concept_count = len([n for n in nodes if n.get("node_type") == "concept"])
        if concept_count > 5:
            insights.append({
                "type": "concept_richness",
                "message": f"Rich conceptual exploration with {concept_count} concepts identified",
                "positive": True
            })
        elif concept_count == 0:
            insights.append({
                "type": "concept_richness",
                "message": "No clear concepts extracted - session may have been too abstract",
                "positive": False
            })
        
        # Metaphor usage insight
        metaphor_count = len([n for n in nodes if n.get("node_type") == "metaphor"])
        if metaphor_count > concept_count and concept_count > 0:
            insights.append({
                "type": "metaphor_usage",
                "message": "Strong metaphorical thinking - good for building intuitive understanding",
                "positive": True
            })
        elif metaphor_count == 0 and concept_count > 2:
            insights.append({
                "type": "metaphor_usage",
                "message": "Limited metaphor usage - could benefit from more analogical thinking",
                "positive": False
            })
        
        # Connection insight
        if len(edges) > len(nodes):
            insights.append({
                "type": "connectivity",
                "message": "High connectivity - concepts are well integrated",
                "positive": True
            })
        elif len(edges) == 0 and len(nodes) > 1:
            insights.append({
                "type": "connectivity",
                "message": "Concepts appear isolated - work on connecting ideas",
                "positive": False
            })
        
        return insights
    
    async def get_learning_trajectory(self, days_back: int = 30) -> Dict[str, Any]:
        """Get learning trajectory over time."""
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)
        
        query = """
        MATCH (s:Session)
        WHERE datetime(s.start_time) >= datetime($start_date)
        AND datetime(s.start_time) <= datetime($end_date)
        RETURN s
        ORDER BY s.start_time
        """
        
        with self.db.session() as session:
            result = session.run(query, {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            })
            sessions = [dict(record["s"]) for record in result]
        
        # Analyze trajectory
        if not sessions:
            return {"message": "No sessions found in the specified period"}
        
        # Calculate metrics over time
        daily_stats = {}
        total_concepts = 0
        total_connections = 0
        
        for session_data in sessions:
            session_date = datetime.fromisoformat(session_data["start_time"]).date()
            date_key = session_date.isoformat()
            
            if date_key not in daily_stats:
                daily_stats[date_key] = {
                    "sessions": 0,
                    "concepts": 0,
                    "connections": 0,
                    "topics": set()
                }
            
            daily_stats[date_key]["sessions"] += 1
            daily_stats[date_key]["concepts"] += len(session_data.get("nodes_created", []))
            daily_stats[date_key]["connections"] += len(session_data.get("edges_created", []))
            daily_stats[date_key]["topics"].add(session_data.get("topic", "unknown"))
            
            total_concepts += len(session_data.get("nodes_created", []))
            total_connections += len(session_data.get("edges_created", []))
        
        # Convert sets to lists for JSON serialization
        for date_key in daily_stats:
            daily_stats[date_key]["topics"] = list(daily_stats[date_key]["topics"])
        
        return {
            "period": {
                "start_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "days": days_back
            },
            "summary": {
                "total_sessions": len(sessions),
                "total_concepts": total_concepts,
                "total_connections": total_connections,
                "avg_concepts_per_session": round(total_concepts / len(sessions), 1),
                "avg_connections_per_session": round(total_connections / len(sessions), 1)
            },
            "daily_stats": daily_stats
        }