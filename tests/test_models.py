"""Test data models."""

import pytest
from datetime import datetime
from peengine.models.graph import Node, Edge, Session, ConceptExtraction, Vector, NodeType, EdgeType


def test_node_creation():
    """Test Node model creation."""
    node = Node(
        id="test_node_1",
        label="Test Concept",
        node_type=NodeType.CONCEPT,
        properties={"domain": "test"}
    )
    
    assert node.id == "test_node_1"
    assert node.label == "Test Concept"
    assert node.node_type == NodeType.CONCEPT
    assert node.properties["domain"] == "test"
    assert isinstance(node.created_at, datetime)


def test_edge_creation():
    """Test Edge model creation."""
    edge = Edge(
        id="test_edge_1",
        source_id="node_1",
        target_id="node_2", 
        edge_type=EdgeType.METAPHORICAL,
        weight=0.8
    )
    
    assert edge.id == "test_edge_1"
    assert edge.source_id == "node_1"
    assert edge.target_id == "node_2"
    assert edge.edge_type == EdgeType.METAPHORICAL
    assert edge.weight == 0.8
    assert isinstance(edge.created_at, datetime)


def test_session_creation():
    """Test Session model creation."""
    session = Session(
        id="test_session_1",
        title="Test Session",
        topic="Test Topic"
    )
    
    assert session.id == "test_session_1"
    assert session.title == "Test Session" 
    assert session.topic == "Test Topic"
    assert session.status == "active"
    assert isinstance(session.start_time, datetime)
    assert session.end_time is None


def test_concept_extraction():
    """Test ConceptExtraction model."""
    extraction = ConceptExtraction(
        concept="quantum entanglement",
        domain="physics",
        metaphors=["spooky action", "dancing partners"],
        confidence=0.9,
        context="Discussion about quantum mechanics"
    )
    
    assert extraction.concept == "quantum entanglement"
    assert extraction.domain == "physics"
    assert len(extraction.metaphors) == 2
    assert extraction.confidence == 0.9


def test_vector_model():
    """Test Vector model."""
    vector = Vector(
        values=[0.1, 0.2, 0.3, 0.4],
        model="test-embedding",
        dimension=4
    )
    
    assert len(vector.values) == 4
    assert vector.model == "test-embedding"
    assert vector.dimension == 4