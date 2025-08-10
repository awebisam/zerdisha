"""Import existing knowledge graphs from JSON files into Neo4j."""

import json
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..models.config import Settings
from ..database.neo4j_client import Neo4jClient
from ..models.graph import Node, Edge, NodeType, EdgeType

logger = logging.getLogger(__name__)


def serialize_property(value: Any) -> Any:
    """Serialize property values to Neo4j-compatible primitive types."""
    if isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, list):
        # Check if all items are primitives
        if all(isinstance(item, (str, int, float, bool)) for item in value):
            return value
        else:
            # Serialize complex list to JSON string
            return json.dumps(value)
    elif isinstance(value, dict):
        # Serialize dict to JSON string
        return json.dumps(value)
    else:
        # Convert other types to string
        return str(value)


class KnowledgeGraphImporter:
    """Import existing JSON knowledge graphs into Neo4j."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = Neo4jClient(settings.database_config)
        self.imported_nodes = {}  # Track imported nodes by original ID
        self.imported_edges = {}  # Track imported edges
        
    async def import_all_graphs(self) -> Dict[str, Any]:
        """Import all knowledge graphs from the knowledge_graphs directory."""
        
        # Connect to database
        self.db.connect()
        self.db.create_indexes()
        
        knowledge_graphs_path = Path(self.settings.knowledge_graphs_path)
        results = {
            "domains_imported": 0,
            "connections_imported": 0,
            "total_nodes": 0,
            "total_edges": 0,
            "errors": []
        }
        
        try:
            # Import domain-specific graphs
            domains_path = knowledge_graphs_path / "domains"
            if domains_path.exists():
                for json_file in domains_path.glob("*.json"):
                    try:
                        result = await self._import_domain_graph(json_file)
                        results["domains_imported"] += 1
                        results["total_nodes"] += result["nodes"]
                        results["total_edges"] += result["edges"]
                        logger.info(f"Imported domain: {json_file.stem}")
                    except Exception as e:
                        error_msg = f"Failed to import {json_file.name}: {e}"
                        results["errors"].append(error_msg)
                        logger.error(error_msg)
            
            # Import connection graphs
            connections_path = knowledge_graphs_path / "connections"
            if connections_path.exists():
                for json_file in connections_path.glob("*.json"):
                    try:
                        result = await self._import_connection_graph(json_file)
                        results["connections_imported"] += 1
                        results["total_nodes"] += result["nodes"]  # Some connection files may have nodes
                        results["total_edges"] += result["edges"]
                        logger.info(f"Imported connections: {json_file.stem}")
                    except Exception as e:
                        error_msg = f"Failed to import {json_file.name}: {e}"
                        results["errors"].append(error_msg)
                        logger.error(error_msg)
            
            return results
            
        finally:
            self.db.close()
    
    async def _import_domain_graph(self, json_file: Path) -> Dict[str, int]:
        """Import a domain-specific knowledge graph."""
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        domain = json_file.stem
        nodes_created = 0
        edges_created = 0
        
        # Handle different JSON structures
        if isinstance(data, dict):
            # Structure: {"nodes": [...], "edges": [...]} or {"nodes": [...], "relationships": [...]} or direct concept mapping
            if "nodes" in data and ("edges" in data or "relationships" in data):
                # Standard graph structure
                nodes_created = await self._import_nodes(data["nodes"], domain)
                relationships = data.get("edges", data.get("relationships", []))
                edges_created = await self._import_edges(relationships)
            else:
                # Concept mapping structure (like your existing files)
                nodes_created = await self._import_concept_mapping(data, domain)
        
        elif isinstance(data, list):
            # List of concepts/nodes
            nodes_created = await self._import_concept_list(data, domain)
        
        return {"nodes": nodes_created, "edges": edges_created}
    
    async def _import_connection_graph(self, json_file: Path) -> Dict[str, int]:
        """Import metaphor connections or cross-domain links."""
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if this is actually a domain graph file with nodes and relationships
        if isinstance(data, dict) and "nodes" in data and ("relationships" in data or "edges" in data):
            # Process as domain graph instead
            domain = json_file.stem
            nodes_created = await self._import_nodes(data["nodes"], domain)
            relationships = data.get("edges", data.get("relationships", []))
            edges_created = await self._import_edges(relationships)
            return {"nodes": nodes_created, "edges": edges_created}
        
        # Regular connection processing
        edges_created = 0
        
        if isinstance(data, dict):
            if "connections" in data:
                edges_created = await self._import_metaphor_connections(data["connections"])
            elif "metaphor_connections" in data:
                edges_created = await self._import_metaphor_connections(data["metaphor_connections"])
            elif "metaphor_bridges" in data:
                edges_created = await self._import_metaphor_bridges(data["metaphor_bridges"])
            else:
                # Direct connection mapping
                edges_created = await self._import_connection_mapping(data)
        
        elif isinstance(data, list):
            # List of connections
            edges_created = await self._import_connection_list(data)
        
        return {"nodes": 0, "edges": edges_created}
    
    async def _import_concept_mapping(self, data: Dict[str, Any], domain: str) -> int:
        """Import concept mapping structure from your existing files."""
        nodes_created = 0
        
        for concept_key, concept_data in data.items():
            if isinstance(concept_data, dict):
                # Create concept node
                node_id = str(uuid.uuid4())
                
                # Serialize all properties to Neo4j-compatible types
                raw_properties = {
                    "domain": domain,
                    "original_key": concept_key,
                    "description": concept_data.get("description", ""),
                    "understanding_level": concept_data.get("understanding_level", ""),
                    "metaphors": concept_data.get("metaphors", []),
                    "related_concepts": concept_data.get("related_concepts", []),
                    "applications": concept_data.get("applications", []),
                    "imported_from": str(json_file.name),
                    "import_date": datetime.utcnow().isoformat()
                }
                
                serialized_properties = {k: serialize_property(v) for k, v in raw_properties.items()}
                
                node = Node(
                    id=node_id,
                    label=concept_data.get("name", concept_key),
                    node_type=NodeType.CONCEPT,
                    properties=serialized_properties
                )
                
                if self.db.create_node(node):
                    self.imported_nodes[f"{domain}:{concept_key}"] = node_id
                    nodes_created += 1
                
                # Create metaphor nodes if they exist
                for metaphor in concept_data.get("metaphors", []):
                    metaphor_node_id = str(uuid.uuid4())
                    metaphor_node = Node(
                        id=metaphor_node_id,
                        label=metaphor,
                        node_type=NodeType.METAPHOR,
                        properties={
                            "domain": domain,
                            "concept": concept_data.get("name", concept_key),
                            "imported_from": str(json_file.name)
                        }
                    )
                    
                    if self.db.create_node(metaphor_node):
                        # Create edge between concept and metaphor
                        edge = Edge(
                            id=str(uuid.uuid4()),
                            source_id=node_id,
                            target_id=metaphor_node_id,
                            edge_type=EdgeType.METAPHORICAL,
                            properties={"imported": True}
                        )
                        self.db.create_edge(edge)
        
        return nodes_created
    
    async def _import_metaphor_connections(self, connections: List[Dict[str, Any]]) -> int:
        """Import metaphor connections between concepts."""
        edges_created = 0
        
        for connection in connections:
            if isinstance(connection, dict):
                source_concept = connection.get("source_concept", "")
                target_concept = connection.get("target_concept", "")
                metaphor = connection.get("metaphor", "")
                
                if source_concept and target_concept:
                    # Find corresponding nodes
                    source_node_id = self._find_concept_node(source_concept)
                    target_node_id = self._find_concept_node(target_concept)
                    
                    if source_node_id and target_node_id:
                        edge = Edge(
                            id=str(uuid.uuid4()),
                            source_id=source_node_id,
                            target_id=target_node_id,
                            edge_type=EdgeType.METAPHORICAL,
                            properties={
                                "metaphor": metaphor,
                                "domains": connection.get("domains", []),
                                "strength": connection.get("strength", 1.0),
                                "bidirectional": connection.get("bidirectional", True),
                                "imported": True
                            }
                        )
                        
                        if self.db.create_edge(edge):
                            edges_created += 1
        
        return edges_created
    
    def _find_concept_node(self, concept_name: str) -> Optional[str]:
        """Find a concept node by name in the database."""
        query = """
        MATCH (n:Node {node_type: 'concept'})
        WHERE toLower(n.label) = toLower($concept_name)
        RETURN n.id as node_id
        LIMIT 1
        """
        
        with self.db.session() as session:
            result = session.run(query, {"concept_name": concept_name})
            record = result.single()
            return record["node_id"] if record else None
    
    async def _import_nodes(self, nodes: List[Dict[str, Any]], domain: str) -> int:
        """Import standard node list."""
        nodes_created = 0
        
        for node_data in nodes:
            node_id = str(uuid.uuid4())
            
            # Handle different node formats
            properties = node_data.get("properties", {})
            node_name = (
                properties.get("name") or 
                node_data.get("label") or 
                node_data.get("name") or 
                "Unknown"
            )
            
            # Determine node type from labels or properties
            labels = node_data.get("labels", [])
            if "Metaphor" in labels:
                node_type = NodeType.METAPHOR
            elif "Concept" in labels or properties.get("type") == "foundational_metaphor":
                node_type = NodeType.CONCEPT
            else:
                node_type = NodeType.CONCEPT  # Default
            
            # Serialize all properties to Neo4j-compatible types
            raw_properties = {
                "domain": domain,
                "original_labels": labels,
                **properties,
                "imported": True,
                "import_date": datetime.utcnow().isoformat()
            }
            
            serialized_properties = {k: serialize_property(v) for k, v in raw_properties.items()}
            
            node = Node(
                id=node_id,
                label=node_name,
                node_type=node_type,
                properties=serialized_properties
            )
            
            if self.db.create_node(node):
                # Store by original ID for relationship mapping
                original_id = node_data.get("id", node_id)
                self.imported_nodes[original_id] = node_id
                nodes_created += 1
        
        return nodes_created
    
    async def _import_edges(self, edges: List[Dict[str, Any]]) -> int:
        """Import standard edge list."""
        edges_created = 0
        
        for edge_data in edges:
            # Handle both formats: source/target and startNode/endNode
            source_key = edge_data.get("source") or edge_data.get("startNode")
            target_key = edge_data.get("target") or edge_data.get("endNode")
            
            source_id = self.imported_nodes.get(source_key)
            target_id = self.imported_nodes.get(target_key)
            
            if source_id and target_id:
                edge_type = edge_data.get("type", "canonical").lower()
                # Map relationship types to our EdgeType enum
                if edge_type in ["evolved_into", "refined_into", "leads_to"]:
                    edge_type_enum = EdgeType.BUILDS_ON
                elif edge_type in ["supports", "enables", "grounds_in"]:
                    edge_type_enum = EdgeType.CANONICAL
                elif edge_type in ["manifests_as", "crystallizes_as"]:
                    edge_type_enum = EdgeType.METAPHORICAL
                else:
                    edge_type_enum = EdgeType.CANONICAL
                
                edge = Edge(
                    id=str(uuid.uuid4()),
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_type_enum,
                    properties={
                        **edge_data.get("properties", {}),
                        "original_type": edge_type,
                        "imported": True
                    }
                )
                
                if self.db.create_edge(edge):
                    edges_created += 1
        
        return edges_created
    
    async def _import_metaphor_bridges(self, bridges: List[Dict[str, Any]]) -> int:
        """Import metaphor bridges between concepts."""
        edges_created = 0
        
        for bridge in bridges:
            if isinstance(bridge, dict):
                concept1_id = bridge.get("concept1_id", "")
                concept2_id = bridge.get("concept2_id", "")
                
                if concept1_id and concept2_id:
                    # Find corresponding nodes by concept ID
                    source_node_id = self._find_concept_by_id(concept1_id)
                    target_node_id = self._find_concept_by_id(concept2_id)
                    
                    if source_node_id and target_node_id:
                        # Serialize properties
                        raw_properties = {
                            "metaphor_type": bridge.get("metaphor_type", ""),
                            "description": bridge.get("description", ""),
                            "strength": bridge.get("strength", 1.0),
                            "domain1": bridge.get("domain1", ""),
                            "domain2": bridge.get("domain2", ""),
                            "imported": True
                        }
                        
                        serialized_properties = {k: serialize_property(v) for k, v in raw_properties.items()}
                        
                        edge = Edge(
                            id=str(uuid.uuid4()),
                            source_id=source_node_id,
                            target_id=target_node_id,
                            edge_type=EdgeType.METAPHORICAL,
                            properties=serialized_properties
                        )
                        
                        if self.db.create_edge(edge):
                            edges_created += 1
        
        return edges_created
    
    def _find_concept_by_id(self, concept_id: str) -> Optional[str]:
        """Find a concept node by its original concept ID."""
        # First check our imported nodes mapping
        for key, node_id in self.imported_nodes.items():
            if concept_id in key:
                return node_id
        
        # If not found, search in database
        query = """
        MATCH (n:Node {node_type: 'concept'})
        WHERE n.original_key = $concept_id OR n.label CONTAINS $concept_id
        RETURN n.id as node_id
        LIMIT 1
        """
        
        with self.db.session() as session:
            result = session.run(query, {"concept_id": concept_id})
            record = result.single()
            return record["node_id"] if record else None
    
    async def _import_concept_list(self, concepts: List[Dict[str, Any]], domain: str) -> int:
        """Import a list of concept objects."""
        nodes_created = 0
        
        for concept in concepts:
            node_id = str(uuid.uuid4())
            
            # Serialize all properties to Neo4j-compatible types
            raw_properties = {
                "domain": domain,
                **concept,
                "imported": True,
                "import_date": datetime.utcnow().isoformat()
            }
            
            serialized_properties = {k: serialize_property(v) for k, v in raw_properties.items()}
            
            node = Node(
                id=node_id,
                label=concept.get("name", concept.get("label", "Unknown")),
                node_type=NodeType.CONCEPT,
                properties=serialized_properties
            )
            
            if self.db.create_node(node):
                nodes_created += 1
        
        return nodes_created
    
    async def _import_connection_mapping(self, data: Dict[str, Any]) -> int:
        """Import direct connection mapping."""
        edges_created = 0
        
        for source_concept, connections in data.items():
            if isinstance(connections, list):
                for target_concept in connections:
                    source_node_id = self._find_concept_node(source_concept)
                    target_node_id = self._find_concept_node(target_concept)
                    
                    if source_node_id and target_node_id:
                        edge = Edge(
                            id=str(uuid.uuid4()),
                            source_id=source_node_id,
                            target_id=target_node_id,
                            edge_type=EdgeType.CANONICAL,
                            properties={"imported": True}
                        )
                        
                        if self.db.create_edge(edge):
                            edges_created += 1
        
        return edges_created
    
    async def _import_connection_list(self, connections: List[Dict[str, Any]]) -> int:
        """Import list of connection objects."""
        edges_created = 0
        
        for connection in connections:
            source_concept = connection.get("source", connection.get("from"))
            target_concept = connection.get("target", connection.get("to"))
            
            if source_concept and target_concept:
                source_node_id = self._find_concept_node(source_concept)
                target_node_id = self._find_concept_node(target_concept)
                
                if source_node_id and target_node_id:
                    edge = Edge(
                        id=str(uuid.uuid4()),
                        source_id=source_node_id,
                        target_id=target_node_id,
                        edge_type=EdgeType.CANONICAL,
                        properties={
                            **connection.get("properties", {}),
                            "imported": True
                        }
                    )
                    
                    if self.db.create_edge(edge):
                        edges_created += 1
        
        return edges_created


async def main():
    """Main import function."""
    settings = Settings()
    importer = KnowledgeGraphImporter(settings)
    
    print("🧠 Importing existing knowledge graphs...")
    results = await importer.import_all_graphs()
    
    print(f"✅ Import complete!")
    print(f"   Domains imported: {results['domains_imported']}")
    print(f"   Connections imported: {results['connections_imported']}")
    print(f"   Total nodes: {results['total_nodes']}")
    print(f"   Total edges: {results['total_edges']}")
    
    if results['errors']:
        print(f"⚠️  Errors encountered:")
        for error in results['errors']:
            print(f"   • {error}")


if __name__ == "__main__":
    asyncio.run(main())