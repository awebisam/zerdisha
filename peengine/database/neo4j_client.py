"""Neo4j database client and operations."""

import logging
from typing import List, Dict, Optional, Any
from neo4j import GraphDatabase, Driver, Session as Neo4jSession
from contextlib import contextmanager

from ..models.graph import Node, Edge, Session, Vector
from ..models.config import DatabaseConfig

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j database client for graph operations."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._driver: Optional[Driver] = None
    
    def connect(self) -> None:
        """Connect to Neo4j database."""
        try:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password)
            )
            # Test connection
            with self._driver.session(database=self.config.database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info("Connected to Neo4j database")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self) -> None:
        """Close database connection."""
        if self._driver:
            self._driver.close()
            logger.info("Closed Neo4j connection")
    
    @contextmanager
    def session(self):
        """Context manager for Neo4j sessions."""
        if not self._driver:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        session = self._driver.session(database=self.config.database)
        try:
            yield session
        finally:
            session.close()
    
    def create_indexes(self) -> None:
        """Create necessary indexes for performance."""
        indexes = [
            "CREATE INDEX node_id_index IF NOT EXISTS FOR (n:Node) ON (n.id)",
            "CREATE INDEX node_type_index IF NOT EXISTS FOR (n:Node) ON (n.node_type)",
            "CREATE INDEX session_id_index IF NOT EXISTS FOR (s:Session) ON (s.id)",
            "CREATE INDEX edge_type_index IF NOT EXISTS FOR ()-[r:RELATES]-() ON (r.edge_type)",
        ]
        
        with self.session() as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                    logger.info(f"Created index: {index_query}")
                except Exception as e:
                    logger.warning(f"Index creation failed (may already exist): {e}")
    
    def create_node(self, node: Node) -> bool:
        """Create or update a node in the graph (idempotent using MERGE)."""
        # Flatten properties into the main node structure
        flattened_properties = {
            "id": node.id,
            "label": node.label,
            "node_type": node.node_type.value,
            "u_vector": node.u_vector.model_dump() if node.u_vector else None,
            "c_vector": node.c_vector.model_dump() if node.c_vector else None,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
            **node.properties  # Flatten custom properties directly into node
        }
        
        # Use MERGE for idempotency - match on ID, then set all properties
        set_assignments = ", ".join([f"n.{key} = ${key}" for key in flattened_properties.keys()])
        query = f"MERGE (n:Node {{id: $id}}) SET {set_assignments}"
        
        with self.session() as session:
            try:
                result = session.run(query, flattened_properties)
                logger.info(f"Merged node: {node.id}")
                return True
            except Exception as e:
                logger.error(f"Failed to merge node {node.id}: {e}")
                return False
    
    def create_edge(self, edge: Edge) -> bool:
        """Create or update an edge between nodes (idempotent using MERGE)."""
        # Flatten properties into the main edge structure
        flattened_properties = {
            "id": edge.id,
            "edge_type": edge.edge_type.value,
            "weight": edge.weight,
            "confidence": edge.confidence,
            "created_at": edge.created_at.isoformat(),
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            **edge.properties  # Flatten custom properties directly into edge
        }
        
        # Use MERGE for idempotency - match on ID and endpoints, then set all properties
        edge_set_assignments = ", ".join([f"r.{key} = ${key}" for key in flattened_properties.keys() if key not in ["source_id", "target_id"]])
        query = f"""
        MATCH (source:Node {{id: $source_id}})
        MATCH (target:Node {{id: $target_id}})
        MERGE (source)-[r:RELATES {{id: $id}}]->(target)
        SET {edge_set_assignments}
        """
        
        with self.session() as session:
            try:
                result = session.run(query, flattened_properties)
                logger.info(f"Merged edge: {edge.id}")
                return True
            except Exception as e:
                logger.error(f"Failed to merge edge {edge.id}: {e}")
                return False
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        query = "MATCH (n:Node {id: $id}) RETURN n"
        
        with self.session() as session:
            result = session.run(query, {"id": node_id})
            record = result.single()
            if record:
                return dict(record["n"])
            return None
    
    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Get all nodes of a specific type."""
        query = "MATCH (n:Node {node_type: $node_type}) RETURN n"
        
        with self.session() as session:
            result = session.run(query, {"node_type": node_type})
            return [dict(record["n"]) for record in result]
    
    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        """Get an edge by ID with source and target node information."""
        query = """
        MATCH (source:Node)-[r:RELATES {id: $edge_id}]->(target:Node)
        RETURN r, source.id as source_id, source.label as source_label, 
               target.id as target_id, target.label as target_label
        """
        
        with self.session() as session:
            result = session.run(query, {"edge_id": edge_id})
            record = result.single()
            if record:
                edge_data = dict(record["r"])
                edge_data.update({
                    "source_id": record["source_id"],
                    "source_label": record["source_label"],
                    "target_id": record["target_id"],
                    "target_label": record["target_label"]
                })
                return edge_data
            return None
    
    def get_connected_nodes(self, node_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get nodes connected to a given node within max_depth."""
        query = """
        MATCH (start:Node {id: $node_id})
        MATCH (start)-[*1..$max_depth]-(connected:Node)
        RETURN DISTINCT connected
        """
        
        with self.session() as session:
            result = session.run(query, {"node_id": node_id, "max_depth": max_depth})
            return [dict(record["connected"]) for record in result]
    
    def create_session_summary(self, session: Session) -> bool:
        """Create or update a session summary node in the knowledge graph (idempotent using MERGE)."""
        query = """
        MERGE (s:Session {id: $id})
        SET s.title = $title,
            s.topic = $topic,
            s.start_time = $start_time,
            s.end_time = $end_time,
            s.nodes_created = $nodes_created,
            s.edges_created = $edges_created,
            s.status = $status,
            s.concept_count = $concept_count,
            s.exchange_count = $exchange_count
        """
        
        with self.session() as session_db:
            try:
                result = session_db.run(query, {
                    "id": session.id,
                    "title": session.title,
                    "topic": session.topic,
                    "start_time": session.start_time.isoformat(),
                    "end_time": session.end_time.isoformat() if session.end_time else None,
                    "nodes_created": session.nodes_created,  # List of concept node IDs
                    "edges_created": session.edges_created,  # List of relationship edge IDs
                    "status": session.status,
                    "concept_count": len(session.nodes_created),
                    "exchange_count": len(session.messages)
                })
                logger.info(f"Merged session summary in graph: {session.id}")
                return True
            except Exception as e:
                logger.error(f"Failed to create session summary {session.id}: {e}")
                return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        query = "MATCH (s:Session {id: $id}) RETURN s"
        
        with self.session() as session:
            result = session.run(query, {"id": session_id})
            record = result.single()
            if record:
                return dict(record["s"])
            return None
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update a session with new data."""
        set_clauses = []
        params = {"id": session_id}
        
        for key, value in updates.items():
            set_clauses.append(f"s.{key} = ${key}")
            params[key] = value
        
        query = f"MATCH (s:Session {{id: $id}}) SET {', '.join(set_clauses)}"
        
        with self.session() as session:
            try:
                result = session.run(query, params)
                logger.info(f"Updated session: {session_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to update session {session_id}: {e}")
                return False
    
    def search_nodes_by_content(self, query_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search nodes by text content in properties."""
        query = """
        MATCH (n:Node)
        WHERE toLower(n.label) CONTAINS toLower($query_text)
        RETURN n
        LIMIT $limit
        """
        
        with self.session() as session:
            result = session.run(query, {"query_text": query_text, "limit": limit})
            return [dict(record["n"]) for record in result]
    
    def clear_all_data(self) -> Dict[str, Any]:
        """Clear all data from the Neo4j database."""
        stats = {"nodes_deleted": 0, "relationships_deleted": 0, "indexes_dropped": 0}
        
        with self.session() as session:
            try:
                # Get counts before deletion
                node_count_result = session.run("MATCH (n) RETURN count(n) as count")
                stats["nodes_deleted"] = node_count_result.single()["count"]
                
                rel_count_result = session.run("MATCH ()-[r]-() RETURN count(r) as count")  
                stats["relationships_deleted"] = rel_count_result.single()["count"]
                
                # Delete all relationships first
                session.run("MATCH ()-[r]-() DELETE r")
                
                # Delete all nodes
                session.run("MATCH (n) DELETE n")
                
                # Drop all indexes and constraints (optional - they'll be recreated on next use)
                try:
                    indexes_result = session.run("SHOW INDEXES")
                    for record in indexes_result:
                        index_name = record.get("name")
                        if index_name:
                            session.run(f"DROP INDEX {index_name} IF EXISTS")
                            stats["indexes_dropped"] += 1
                except Exception as e:
                    logger.warning(f"Could not drop indexes: {e}")
                
                logger.info(f"Cleared Neo4j database: {stats['nodes_deleted']} nodes, {stats['relationships_deleted']} relationships")
                return stats
                
            except Exception as e:
                logger.error(f"Failed to clear Neo4j database: {e}")
                raise