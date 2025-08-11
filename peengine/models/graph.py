"""Data models for knowledge graph entities."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    CONCEPT = "concept"
    METAPHOR = "metaphor"
    DOMAIN = "domain"
    SESSION = "session"
    HYPOTHESIS = "hypothesis"
    ARTIFACT = "artifact"


class EdgeType(str, Enum):
    """Types of edges in the knowledge graph."""
    METAPHORICAL = "metaphorical"
    CANONICAL = "canonical"
    BUILDS_ON = "builds_on"
    CONTRADICTS = "contradicts"
    EXPLORES = "explores"
    GENERATED_BY = "generated_by"


class Vector(BaseModel):
    """Vector representation for embeddings."""
    values: List[float]
    model: str = "text-embedding-ada-002"
    dimension: int = Field(default_factory=lambda: 1536)
    metadata: Dict[str, Any] = Field(
        default_factory=dict)  # Store additional data


class Node(BaseModel):
    """A node in the knowledge graph with rich semantic label support."""
    id: str
    label: str  # Display name
    node_type: NodeType  # Primary type
    # Rich Neo4j labels like ["Concept", "EarlyInsight"]
    labels: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None  # Every node may belong to a session
    properties: Dict[str, Any] = Field(default_factory=dict)
    u_vector: Optional[Vector] = None  # User's metaphorical understanding
    c_vector: Optional[Vector] = None  # Canonical academic vector
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def get_neo4j_labels(self) -> List[str]:
        """Get the full list of labels for Neo4j, combining primary type and rich labels."""
        # Start with base labels
        base_labels = ["Node"]

        # Add rich semantic labels if provided
        if self.labels:
            base_labels.extend(self.labels)
        else:
            # Fallback to node_type based label
            base_labels.append(self.node_type.value.title())

        return base_labels

    def get_labels_string(self) -> str:
        """Get labels formatted for Cypher queries like 'Node:Concept:EarlyInsight'."""
        return ":".join(self.get_neo4j_labels())

    def is_intrasession(self, other_node: 'Node') -> bool:
        """Check if this node is in the same session as another node."""
        return self.session_id == other_node.session_id


class Edge(BaseModel):
    """An edge in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    # Intrasession edges have session_id, cross-session don't
    session_id: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExplorationSession(BaseModel):
    """A knowledge exploration session - first-class graph entity."""
    id: str
    domain: str
    topic: str
    session_type: str = "exploration"  # exploration, meta_learning, live
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_minutes: Optional[float] = None
    node_count: int = 0
    edge_count: int = 0
    breakthrough_count: int = 0
    properties: Dict[str, Any] = Field(default_factory=dict)


class LiveSession(BaseModel):
    """Active learning session for real-time exploration."""
    id: str
    exploration_session_id: str  # Links to ExplorationSession
    title: str
    topic: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    nodes_created: List[str] = Field(default_factory=list)
    edges_created: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    status: str = "active"  # active, completed, paused


class Session(BaseModel):
    """Primary session model used across the system and tests.

    Matches expectations from tests and orchestrator:
    - Defaults: status='active', start_time set, end_time=None
    - Tracks messages and created graph artifacts
    """
    id: str
    title: str
    topic: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: str = "active"
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    nodes_created: List[str] = Field(default_factory=list)
    edges_created: List[str] = Field(default_factory=list)
    exploration_session_id: Optional[str] = None


class ConceptExtraction(BaseModel):
    """Extracted concept from conversation."""
    concept: str
    metaphors: List[str] = Field(default_factory=list)
    domain: str
    confidence: float
    context: str
    relationships: List[Dict[str, str]] = Field(default_factory=list)


class MetaphorsConnection(BaseModel):
    """Connection between concepts via metaphor."""
    source_concept: str
    target_concept: str
    metaphor: str
    domains: List[str]
    strength: float
    bidirectional: bool = True


class SeedDiscovery(BaseModel):
    """A seed for new exploration discovered by MA."""
    concept: str
    discovery_type: str
    rationale: str
    related_concepts: List[str] = Field(default_factory=list)
    suggested_questions: List[str] = Field(default_factory=list)
    priority: float = 0.5
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SessionRelation(BaseModel):
    """Cross-session relationships for connecting explorations."""
    id: str
    source_session_id: str
    target_session_id: str
    relation_type: str  # influences, builds_on, contradicts, metaphorically_connects
    metaphor_justification: Optional[str] = None
    confidence: float = 1.0
    created_by: str = "system"  # system, MA, user
    created_at: datetime = Field(default_factory=datetime.utcnow)
