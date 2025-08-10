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


class Node(BaseModel):
    """A node in the knowledge graph."""
    id: str
    label: str
    node_type: NodeType
    properties: Dict[str, Any] = Field(default_factory=dict)
    u_vector: Optional[Vector] = None  # User's metaphorical understanding
    c_vector: Optional[Vector] = None  # Canonical academic vector
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Edge(BaseModel):
    """An edge in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    properties: Dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Session(BaseModel):
    """A learning session."""
    id: str
    title: str
    topic: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    nodes_created: List[str] = Field(default_factory=list)
    edges_created: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    status: str = "active"  # active, completed, paused


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