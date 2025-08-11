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
from ..models.graph import Node, Edge, ExplorationSession, NodeType, EdgeType

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
        self._create_domain_indexes()

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
                        # Some connection files may have nodes
                        results["total_nodes"] += result["nodes"]
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
        """Import a domain-specific knowledge graph as an exploration session."""

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        domain = json_file.stem

        # Create exploration session first
        exploration_session = ExplorationSession(
            id=f"{domain}_session",
            domain=self._get_domain_from_filename(json_file.stem),
            topic=self._infer_topic_from_data(data, domain),
            session_type="imported",
            timestamp=datetime.utcnow()
        )

        nodes_created = 0
        edges_created = 0

        # Handle different JSON structures
        if isinstance(data, dict):
            # Structure: {"nodes": [...], "edges": [...]} or {"nodes": [...], "relationships": [...]} or direct concept mapping
            if "nodes" in data and ("edges" in data or "relationships" in data):
                # Standard graph structure
                nodes_created = await self._import_nodes(data["nodes"], domain, exploration_session)
                relationships = data.get(
                    "edges", data.get("relationships", []))
                edges_created = await self._import_edges(relationships, exploration_session)
            else:
                # Concept mapping structure (like your existing files)
                nodes_created = await self._import_concept_mapping(data, domain, json_file, exploration_session)

        elif isinstance(data, list):
            # List of concepts/nodes
            nodes_created = await self._import_concept_list(data, domain)

        # Update session statistics and create in database
        exploration_session.node_count = nodes_created
        exploration_session.edge_count = edges_created
        exploration_session.breakthrough_count = self._count_breakthroughs(
            data)

        # Create session in graph
        self.db.create_session(exploration_session)

        return {"nodes": nodes_created, "edges": edges_created}

    async def _import_connection_graph(self, json_file: Path) -> Dict[str, int]:
        """Import metaphor connections or cross-domain links."""

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if this is actually a domain graph file with nodes and relationships
        if isinstance(data, dict) and "nodes" in data and ("relationships" in data or "edges" in data):
            # This connection file is actually a graph with nodes+edges; treat it as a session import
            domain = json_file.stem
            exploration_session = ExplorationSession(
                id=f"{domain}_session",
                domain=self._get_domain_from_filename(json_file.stem),
                topic=self._infer_topic_from_data(data, domain),
                session_type="imported",
                timestamp=datetime.utcnow()
            )

            nodes_created = await self._import_nodes(data["nodes"], domain, exploration_session)
            relationships = data.get("edges", data.get("relationships", []))
            edges_created = await self._import_edges(relationships, exploration_session)

            # Update and persist exploration session container
            exploration_session.node_count = nodes_created
            exploration_session.edge_count = edges_created
            exploration_session.breakthrough_count = self._count_breakthroughs(
                data)
            self.db.create_session(exploration_session)
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

    async def _import_concept_mapping(self, data: Dict[str, Any], domain: str, json_file: Path, exploration_session: ExplorationSession) -> int:
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

                serialized_properties = {k: serialize_property(
                    v) for k, v in raw_properties.items()}

                # Create concept mapping with rich labels
                concept_id = f"{domain}:{concept_key}"

                # Use Node model with Concept label for concept mapping format
                node = Node(
                    id=concept_id,
                    label=concept_data.get("name", concept_key),
                    node_type=NodeType.CONCEPT,
                    # Default rich label for concept mapping
                    labels=["Concept"],
                    session_id=f"{domain}_session",  # Assign to session
                    properties={**serialized_properties,
                                "concept_id": concept_id}
                )

                if self.db.create_node(node):
                    self.imported_nodes[concept_key] = concept_id
                    self.imported_nodes[f"{domain}:{concept_key}"] = concept_id
                    nodes_created += 1
                else:
                    logger.error(
                        f"Failed to create concept mapping node {concept_key}")

                # Create metaphor nodes if they exist
                for metaphor in concept_data.get("metaphors", []):
                    metaphor_node_id = str(uuid.uuid4())
                    metaphor_id = f"{domain}:metaphor:{metaphor}"
                    metaphor_properties = {
                        "concept_id": metaphor_id,
                        "domain": domain,
                        "concept": concept_data.get("name", concept_key),
                        "imported": True
                    }

                    # Create metaphor node with rich labels using Node model
                    metaphor_node = Node(
                        id=metaphor_id,
                        label=metaphor,
                        node_type=NodeType.METAPHOR,
                        # Rich labels for metaphors
                        labels=["Concept", "Metaphor"],
                        properties={**metaphor_properties, "name": metaphor}
                    )

                    if self.db.create_node(metaphor_node):
                        # Create metaphorical relationship using Edge model
                        edge = Edge(
                            id=str(uuid.uuid4()),
                            source_id=concept_id,
                            target_id=metaphor_id,
                            edge_type=EdgeType.METAPHORICAL,
                            # Intrasession metaphor edge
                            session_id=f"{domain}_session",
                            properties={"imported": True}
                        )
                        self.db.create_edge(edge)
                    else:
                        logger.error(f"Failed to create metaphor {metaphor}")

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
                        # Create metaphorical relationship using direct Cypher
                        rel_properties = {
                            "metaphor": metaphor,
                            "domains": serialize_property(connection.get("domains", [])),
                            "strength": connection.get("strength", 1.0),
                            "bidirectional": connection.get("bidirectional", True),
                            "imported": True
                        }

                        cypher = """
                        MATCH (start:Concept {concept_id: $start_id})
                        MATCH (end:Concept {concept_id: $end_id})
                        MERGE (start)-[r:METAPHORICAL]->(end)
                        ON CREATE SET r += $rel_properties
                        ON MATCH SET r += $rel_properties
                        """

                        try:
                            with self.db.session() as session:
                                session.run(cypher, {
                                    "start_id": source_node_id,
                                    "end_id": target_node_id,
                                    "rel_properties": rel_properties
                                })
                                edges_created += 1

                        except Exception as e:
                            logger.error(
                                f"Failed to create metaphorical connection: {e}")

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

    async def _import_nodes(self, nodes: List[Dict[str, Any]], domain: str, exploration_session: ExplorationSession) -> int:
        """Import nodes preserving rich semantic structure like reference loader."""
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

            # Preserve original labels as rich Neo4j labels
            original_labels = node_data.get("labels", [])

            # Preserve all properties exactly as they were
            raw_properties = {
                "concept_id": node_data.get("id", node_id),
                "domain": domain,
                "original_labels": original_labels,
                **properties,
                "imported": True,
                "import_date": datetime.utcnow().isoformat()
            }

            serialized_properties = {k: serialize_property(
                v) for k, v in raw_properties.items()}

            # Create Node with rich labels and session assignment
            node = Node(
                id=serialized_properties["concept_id"],
                label=node_name,
                node_type=NodeType.CONCEPT,
                labels=original_labels,  # Rich semantic labels
                session_id=exploration_session.id,
                properties=serialized_properties
            )

            if self.db.create_node(node):
                # Link node to session
                self.db.link_node_to_session(node.id, exploration_session.id)

                # Store by original ID for relationship mapping
                original_id = node_data.get("id", node_id)
                self.imported_nodes[original_id] = serialized_properties["concept_id"]
                nodes_created += 1
                logger.debug(
                    f"Created node with rich labels {original_labels}: {node_name}")
            else:
                logger.error(f"Failed to create node {node_name}")

        return nodes_created

    async def _import_edges(self, edges: List[Dict[str, Any]], exploration_session: ExplorationSession) -> int:
        """Import relationships preserving original semantics like reference loader."""
        edges_created = 0

        for edge_data in edges:
            # Handle both formats: source/target and startNode/endNode
            source_key = edge_data.get("source") or edge_data.get("startNode")
            target_key = edge_data.get("target") or edge_data.get("endNode")

            source_id = self.imported_nodes.get(source_key)
            target_id = self.imported_nodes.get(target_key)

            if source_id and target_id:
                # Use original relationship type as primary type like reference loader
                original_type = edge_data.get("type", "RELATES_TO")
                rel_type = original_type.upper().replace(' ', '_').replace('-', '_')

                # Create intrasession edge using Edge model
                edge = Edge(
                    id=str(uuid.uuid4()),
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=self._map_to_edge_type(original_type),
                    session_id=exploration_session.id,  # Intrasession edge
                    properties={
                        **edge_data.get("properties", {}),
                        "original_type": original_type,
                        "imported": True
                    }
                )

                if self.db.create_edge(edge):
                    edges_created += 1
                    logger.debug(
                        f"Created intrasession relationship {original_type} between {source_id} and {target_id}")
                else:
                    logger.error(
                        f"Failed to create relationship {original_type}")

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

                        serialized_properties = {k: serialize_property(
                            v) for k, v in raw_properties.items()}

                        # Create metaphor bridge using direct Cypher
                        cypher = """
                        MATCH (c1:Concept {concept_id: $concept1_id})
                        MATCH (c2:Concept {concept_id: $concept2_id})
                        MERGE (c1)-[r:METAPHOR_BRIDGE]->(c2)
                        ON CREATE SET r += $rel_properties
                        ON MATCH SET r += $rel_properties
                        """

                        try:
                            with self.db.session() as session:
                                session.run(cypher, {
                                    "concept1_id": source_node_id,
                                    "concept2_id": target_node_id,
                                    "rel_properties": serialized_properties
                                })
                                edges_created += 1

                        except Exception as e:
                            logger.error(
                                f"Failed to create metaphor bridge: {e}")

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

            serialized_properties = {k: serialize_property(
                v) for k, v in raw_properties.items()}

            # Create concept using Node model with rich labels
            concept_id = serialized_properties.get("id", node_id)
            serialized_properties["concept_id"] = concept_id

            node = Node(
                id=concept_id,
                label=concept.get("name", concept.get("label", "Unknown")),
                node_type=NodeType.CONCEPT,
                labels=["Concept"],  # Default rich label
                properties=serialized_properties
            )

            if self.db.create_node(node):
                nodes_created += 1
            else:
                logger.error(
                    f"Failed to create concept list node {concept_id}")

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
                        # Create canonical relationship using direct Cypher
                        cypher = """
                        MATCH (start:Concept {concept_id: $start_id})
                        MATCH (end:Concept {concept_id: $end_id})
                        MERGE (start)-[r:CANONICAL]->(end)
                        ON CREATE SET r.imported = true
                        ON MATCH SET r.imported = true
                        """

                        try:
                            with self.db.session() as session:
                                session.run(cypher, {
                                    "start_id": source_node_id,
                                    "end_id": target_node_id
                                })
                                edges_created += 1

                        except Exception as e:
                            logger.error(
                                f"Failed to create connection mapping edge: {e}")

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
                    # Create connection using direct Cypher
                    rel_properties = {
                        **{k: serialize_property(v) for k, v in connection.get("properties", {}).items()},
                        "imported": True
                    }

                    cypher = """
                    MATCH (start:Concept {concept_id: $start_id})
                    MATCH (end:Concept {concept_id: $end_id})
                    MERGE (start)-[r:CANONICAL]->(end)
                    ON CREATE SET r += $rel_properties
                    ON MATCH SET r += $rel_properties
                    """

                    try:
                        with self.db.session() as session:
                            session.run(cypher, {
                                "start_id": source_node_id,
                                "end_id": target_node_id,
                                "rel_properties": rel_properties
                            })
                            edges_created += 1

                    except Exception as e:
                        logger.error(
                            f"Failed to create connection list edge: {e}")

        return edges_created

    def _create_domain_indexes(self):
        """Create domain-focused indexes like reference loader for optimal querying."""
        indexes = [
            # Core concept indexes - domain-focused
            "CREATE INDEX concept_id_domain IF NOT EXISTS FOR (c:Concept) ON (c.concept_id, c.domain)",
            "CREATE INDEX concept_domain IF NOT EXISTS FOR (c:Concept) ON (c.domain)",
            "CREATE INDEX concept_name IF NOT EXISTS FOR (c:Concept) ON (c.name)",
            "CREATE INDEX concept_type IF NOT EXISTS FOR (c:Concept) ON (c.type)",

            # Support for rich label structures
            "CREATE INDEX concept_original_labels IF NOT EXISTS FOR (c:Concept) ON (c.original_labels)",

            # Full text search on descriptions and names
            "CREATE FULLTEXT INDEX concept_descriptions IF NOT EXISTS FOR (c:Concept) ON [c.description, c.name] OPTIONS {indexConfig: {`fulltext.analyzer`: 'standard'}}"
        ]

        with self.db.session() as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                    logger.debug(f"Created index: {index_query[:50]}...")
                except Exception as e:
                    # Index may already exist, which is fine
                    logger.debug(
                        f"Index creation note (may already exist): {e}")

        logger.info("Created domain-focused indexes for optimal querying")

    def _get_domain_from_filename(self, filename: str) -> str:
        """Extract domain name from filename, handling special cases."""
        domain_mapping = {
            "ai_exploration": "ai",
            "rag": "ai",
            "quantum": "physics",
            "relativity": "physics",
            "vision": "physics",
            "evolution": "biology",
            "chemistry": "biology",
            "philosophy": "philosophy",
            "base": "meta_learning",
            "seed_decoding_discovery": "meta_learning"
        }
        return domain_mapping.get(filename, filename)

    def _infer_topic_from_data(self, data: Dict[str, Any], filename: str) -> str:
        """Infer exploration topic from data content."""
        topic_mapping = {
            "ai_exploration": "Transformers and attention mechanisms",
            "rag": "Retrieval-augmented generation and vector databases",
            "quantum": "Quantum mechanics and measurement paradoxes",
            "relativity": "Spacetime, gravity, and relativistic physics",
            "vision": "Visual perception and light-to-signal processing",
            "evolution": "Natural selection and evolutionary mechanisms",
            "chemistry": "Atomic structure and chemical bonding",
            "philosophy": "Non-duality, systems thinking, and reality deconstruction",
            "base": "Meta-learning methodologies and cognitive tools",
            "seed_decoding_discovery": "Cognitive techniques for learning mode discovery"
        }
        return topic_mapping.get(filename, f"Exploration of {filename}")

    def _count_breakthroughs(self, data: Dict[str, Any]) -> int:
        """Count breakthrough moments in session data."""
        if "nodes" not in data:
            return 0

        breakthrough_count = 0
        for node in data["nodes"]:
            labels = node.get("labels", [])
            if any(label in ["Breakthrough", "EpistemicEvent", "CoreRealization"] for label in labels):
                breakthrough_count += 1
        return breakthrough_count

    def _map_to_edge_type(self, original_type: str) -> EdgeType:
        """Map original relationship type to EdgeType enum."""
        original_lower = original_type.lower()

        if original_lower in ["evolved_into", "refined_into", "leads_to", "built_on"]:
            return EdgeType.BUILDS_ON
        elif original_lower in ["supports", "enables", "grounds_in", "canonical"]:
            return EdgeType.CANONICAL
        elif original_lower in ["manifests_as", "crystallizes_as", "metaphorical"]:
            return EdgeType.METAPHORICAL
        elif original_lower in ["contradicts", "challenges"]:
            return EdgeType.CONTRADICTS
        elif original_lower in ["explores", "investigates"]:
            return EdgeType.EXPLORES
        else:
            return EdgeType.CANONICAL  # Default fallback


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
