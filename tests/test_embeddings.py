"""Test embedding service."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, Mock

from peengine.core.embeddings import EmbeddingService
from peengine.models.config import LLMConfig
from peengine.models.graph import Vector


@pytest.fixture
def llm_config():
    """Test LLM configuration."""
    return LLMConfig(
        azure_openai_key="test_key",
        azure_openai_endpoint="https://test.api.com",
        azure_openai_deployment_name="test-deployment",
        azure_openai_model_name="gpt-4"
    )


@pytest.fixture
def embedding_service(llm_config):
    """Test embedding service."""
    service = EmbeddingService(llm_config)
    # Mock the OpenAI client
    service.client = AsyncMock()
    return service


def test_calculate_similarity():
    """Test vector similarity calculation."""
    config = LLMConfig(
        azure_openai_key="test",
        azure_openai_endpoint="https://test.api.com",
        azure_openai_deployment_name="test-deployment",
        azure_openai_model_name="gpt-4"
    )
    service = EmbeddingService(config)
    
    # Test identical vectors
    vector1 = Vector(values=[1.0, 0.0, 0.0], model="test", dimension=3)
    vector2 = Vector(values=[1.0, 0.0, 0.0], model="test", dimension=3)
    
    similarity = service.calculate_similarity(vector1, vector2)
    assert abs(similarity - 1.0) < 1e-6  # Should be 1.0 for identical vectors
    
    # Test orthogonal vectors
    vector3 = Vector(values=[1.0, 0.0, 0.0], model="test", dimension=3)
    vector4 = Vector(values=[0.0, 1.0, 0.0], model="test", dimension=3)
    
    similarity = service.calculate_similarity(vector3, vector4)
    assert abs(similarity - 0.0) < 1e-6  # Should be 0.0 for orthogonal vectors


def test_calculate_gap_score():
    """Test gap score calculation between vectors."""
    config = LLMConfig(
        azure_openai_key="test",
        azure_openai_endpoint="https://test.api.com",
        azure_openai_deployment_name="test-deployment",
        azure_openai_model_name="gpt-4"
    )
    service = EmbeddingService(config)
    
    # Similar vectors (small gap)
    u_vector = Vector(values=[1.0, 0.1, 0.0], model="test", dimension=3)
    c_vector = Vector(values=[1.0, 0.0, 0.0], model="test", dimension=3)
    
    gap_score = service.calculate_gap_score(u_vector, c_vector)
    
    assert "similarity" in gap_score
    assert "gap_score" in gap_score
    assert "severity" in gap_score
    assert gap_score["gap_score"] == 1.0 - gap_score["similarity"]


def test_find_similar_concepts():
    """Test finding similar concepts."""
    config = LLMConfig(
        azure_openai_key="test",
        azure_openai_endpoint="https://test.api.com",
        azure_openai_deployment_name="test-deployment",
        azure_openai_model_name="gpt-4"
    )
    service = EmbeddingService(config)
    
    target_vector = Vector(values=[1.0, 0.0, 0.0], model="test", dimension=3)
    
    concept_vectors = {
        "concept_a": Vector(values=[1.0, 0.0, 0.0], model="test", dimension=3),  # identical
        "concept_b": Vector(values=[0.9, 0.1, 0.0], model="test", dimension=3),  # similar
        "concept_c": Vector(values=[0.0, 1.0, 0.0], model="test", dimension=3),  # different
    }
    
    similar = service.find_similar_concepts(target_vector, concept_vectors, top_k=2)
    
    assert len(similar) == 2
    assert similar[0][0] == "concept_a"  # Most similar first
    assert similar[0][1] > similar[1][1]  # Decreasing similarity


@pytest.mark.asyncio
async def test_create_u_vector(embedding_service):
    """Test u-vector creation."""
    # Mock embedding response
    embedding_service.client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1, 0.2, 0.3])]
    )
    
    u_vector = await embedding_service.create_u_vector(
        concept="quantum mechanics",
        user_explanation="like spinning coins",
        metaphors=["spinning coins", "dancing particles"]
    )
    
    assert isinstance(u_vector, Vector)
    assert len(u_vector.values) == 3
    assert u_vector.model == "text-embedding-ada-002"


@pytest.mark.asyncio
async def test_create_c_vector(embedding_service):
    """Test c-vector creation."""
    # Mock embedding response
    embedding_service.client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.4, 0.5, 0.6])]
    )
    
    # Mock canonical definition generation
    embedding_service.generate_canonical_definition = AsyncMock(
        return_value="The branch of physics dealing with quantum systems"
    )
    
    c_vector = await embedding_service.create_c_vector(
        concept="quantum mechanics",
        domain="physics"
    )
    
    assert isinstance(c_vector, Vector)
    assert len(c_vector.values) == 3
    assert c_vector.model == "text-embedding-ada-002"
    assert "canonical_definition" in c_vector.metadata
    assert c_vector.metadata["canonical_definition"] == "The branch of physics dealing with quantum systems"
    assert c_vector.metadata["concept"] == "quantum mechanics"
    assert c_vector.metadata["domain"] == "physics"
    assert c_vector.metadata["vector_type"] == "c_vector"


def test_analyze_vector_clusters():
    """Test vector cluster analysis."""
    config = LLMConfig(
        azure_openai_key="test",
        azure_openai_endpoint="https://test.api.com",
        azure_openai_deployment_name="test-deployment",
        azure_openai_model_name="gpt-4"
    )
    service = EmbeddingService(config)
    
    # Create vectors that form a clear cluster
    vectors = {
        "physics_1": Vector(values=[1.0, 0.0, 0.0], model="test", dimension=3),
        "physics_2": Vector(values=[0.9, 0.1, 0.0], model="test", dimension=3),  # Similar to physics_1
        "biology_1": Vector(values=[0.0, 0.0, 1.0], model="test", dimension=3),  # Different
    }
    
    analysis = service.analyze_vector_clusters(vectors)
    
    assert "clusters" in analysis
    assert "analysis" in analysis
    assert isinstance(analysis["clusters"], list)