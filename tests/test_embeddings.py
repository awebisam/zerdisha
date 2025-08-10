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
        data=[Mock(embedding=[0.1] * 1536)]
    )
    
    u_vector = await embedding_service.create_u_vector(
        concept="quantum mechanics",
        user_explanation="like spinning coins",
        metaphors=["spinning coins", "dancing particles"]
    )
    
    assert isinstance(u_vector, Vector)
    assert len(u_vector.values) == 1536
    assert u_vector.model == "text-embedding-ada-002"


@pytest.mark.asyncio
async def test_create_c_vector(embedding_service):
    """Test c-vector creation."""
    # Mock embedding response
    embedding_service.client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.4] * 1536)]
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
    assert len(c_vector.values) == 1536
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


# New tests for embedding service enhancements

@pytest.mark.asyncio
async def test_generate_canonical_definition(embedding_service):
    """Test canonical definition generation with domain-specific prompting."""
    # Mock OpenAI chat completion response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Quantum mechanics is the branch of physics that describes the behavior of matter and energy at the atomic and subatomic scale, characterized by wave-particle duality and probabilistic outcomes."
    
    embedding_service.client.chat.completions.create.return_value = mock_response
    
    # Test physics domain
    definition = await embedding_service.generate_canonical_definition("quantum mechanics", "physics")
    
    assert isinstance(definition, str)
    assert len(definition) > 0
    assert "quantum mechanics" in definition.lower() or "quantum" in definition.lower()
    
    # Verify the API was called with correct parameters
    embedding_service.client.chat.completions.create.assert_called_once()
    call_args = embedding_service.client.chat.completions.create.call_args
    
    assert call_args[1]["model"] == embedding_service.llm_config.azure_openai_deployment_name
    assert call_args[1]["temperature"] == 0.1  # Low temperature for canonical accuracy
    assert call_args[1]["max_tokens"] == 250
    
    # Check that domain-specific guidance is included in the prompt
    prompt = call_args[1]["messages"][0]["content"]
    assert "physics" in prompt.lower()
    assert "quantum mechanics" in prompt
    assert "canonical definition" in prompt.lower()


@pytest.mark.asyncio
async def test_generate_canonical_definition_different_domains(embedding_service):
    """Test canonical definition generation for different domains."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test definition"
    
    embedding_service.client.chat.completions.create.return_value = mock_response
    
    # Test different domains
    domains = ["physics", "chemistry", "biology", "mathematics", "philosophy", "computer_science", "psychology", "economics", "general"]
    
    for domain in domains:
        await embedding_service.generate_canonical_definition("test_concept", domain)
        
        # Check that domain-specific guidance is used
        call_args = embedding_service.client.chat.completions.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        assert domain in prompt.lower()


@pytest.mark.asyncio
async def test_generate_canonical_definition_api_failure(embedding_service):
    """Test canonical definition generation handles API failures gracefully."""
    # Mock API failure
    embedding_service.client.chat.completions.create.side_effect = Exception("API Error")
    
    definition = await embedding_service.generate_canonical_definition("quantum mechanics", "physics")
    
    # Should return fallback definition
    assert isinstance(definition, str)
    assert "quantum mechanics" in definition
    assert "physics" in definition
    assert definition == 'The concept of "quantum mechanics" as understood in the academic domain of physics.'


@pytest.mark.asyncio
async def test_create_c_vector_with_metadata(embedding_service):
    """Test c_vector creation with stored canonical definition."""
    # Mock embedding response
    embedding_service.client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1] * 1536)]
    )
    
    # Mock canonical definition generation
    test_definition = "Quantum mechanics is the fundamental theory in physics that describes nature at the smallest scales."
    embedding_service.generate_canonical_definition = AsyncMock(return_value=test_definition)
    
    c_vector = await embedding_service.create_c_vector("quantum mechanics", "physics")
    
    # Verify vector properties
    assert isinstance(c_vector, Vector)
    assert len(c_vector.values) == 1536
    assert c_vector.model == "text-embedding-ada-002"
    assert c_vector.dimension == 1536
    
    # Verify metadata is properly stored
    assert "canonical_definition" in c_vector.metadata
    assert c_vector.metadata["canonical_definition"] == test_definition
    assert c_vector.metadata["concept"] == "quantum mechanics"
    assert c_vector.metadata["domain"] == "physics"
    assert c_vector.metadata["vector_type"] == "c_vector"
    
    # Verify canonical definition was generated
    embedding_service.generate_canonical_definition.assert_called_once_with("quantum mechanics", "physics")
    
    # Verify embedding was created with proper text
    embedding_service.client.embeddings.create.assert_called_once()
    call_args = embedding_service.client.embeddings.create.call_args
    input_text = call_args[1]["input"]
    assert "quantum mechanics" in input_text
    assert "physics" in input_text
    assert test_definition in input_text
    assert "Canonical definition" in input_text


@pytest.mark.asyncio
async def test_calculate_gap_score_detailed_analysis(embedding_service):
    """Test gap score calculation with detailed analysis and severity categorization."""
    # Create test vectors with known similarity
    u_vector = Vector(
        values=[1.0, 0.0, 0.0, 0.0, 0.0],
        model="test",
        dimension=5
    )
    
    c_vector = Vector(
        values=[0.8, 0.6, 0.0, 0.0, 0.0],  # Will have specific similarity
        model="test",
        dimension=5,
        metadata={"canonical_definition": "Test canonical definition"}
    )
    
    gap_analysis = embedding_service.calculate_gap_score(u_vector, c_vector)
    
    # Verify all required fields are present
    required_fields = [
        "similarity", "gap_score", "severity", "severity_description",
        "u_vector_magnitude", "c_vector_magnitude", "emphasis_difference",
        "canonical_definition", "analysis_timestamp"
    ]
    
    for field in required_fields:
        assert field in gap_analysis, f"Missing field: {field}"
    
    # Verify calculations
    assert isinstance(gap_analysis["similarity"], float)
    assert isinstance(gap_analysis["gap_score"], float)
    assert gap_analysis["gap_score"] == 1.0 - gap_analysis["similarity"]
    
    # Verify severity classification
    assert gap_analysis["severity"] in ["minimal", "low", "moderate", "high", "critical"]
    assert isinstance(gap_analysis["severity_description"], str)
    
    # Verify magnitude calculations
    expected_u_magnitude = np.linalg.norm([1.0, 0.0, 0.0, 0.0, 0.0])
    expected_c_magnitude = np.linalg.norm([0.8, 0.6, 0.0, 0.0, 0.0])
    
    assert abs(gap_analysis["u_vector_magnitude"] - expected_u_magnitude) < 1e-6
    assert abs(gap_analysis["c_vector_magnitude"] - expected_c_magnitude) < 1e-6
    
    # Verify emphasis difference calculation
    expected_emphasis_diff = abs(expected_u_magnitude - expected_c_magnitude) / expected_c_magnitude
    assert abs(gap_analysis["emphasis_difference"] - expected_emphasis_diff) < 1e-6
    
    # Verify metadata inclusion
    assert gap_analysis["canonical_definition"] == "Test canonical definition"
    assert "T" in gap_analysis["analysis_timestamp"]  # ISO format check


def test_calculate_gap_score_severity_categories():
    """Test gap score severity categorization for different similarity levels."""
    config = LLMConfig(
        azure_openai_key="test",
        azure_openai_endpoint="https://test.api.com",
        azure_openai_deployment_name="test-deployment",
        azure_openai_model_name="gpt-4"
    )
    service = EmbeddingService(config)
    
    # Test different similarity levels and their severity classifications
    test_cases = [
        ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], "minimal"),  # Perfect similarity (1.0)
        ([1.0, 0.0, 0.0], [0.9, 0.1, 0.0], "minimal"),  # High similarity (~0.95)
        ([1.0, 0.0, 0.0], [0.8, 0.6, 0.0], "low"),      # Good similarity (~0.8)
        ([1.0, 0.0, 0.0], [0.6, 0.8, 0.0], "moderate"), # Moderate similarity (~0.6)
        ([1.0, 0.0, 0.0], [0.4, 0.9, 0.0], "high"),     # Low similarity (~0.4)
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], "critical"), # No similarity (0.0)
    ]
    
    for u_vals, c_vals, expected_severity in test_cases:
        u_vector = Vector(values=u_vals, model="test", dimension=3)
        c_vector = Vector(values=c_vals, model="test", dimension=3, metadata={"canonical_definition": "test"})
        
        gap_analysis = service.calculate_gap_score(u_vector, c_vector)
        assert gap_analysis["severity"] == expected_severity, f"Expected {expected_severity}, got {gap_analysis['severity']} for similarity {gap_analysis['similarity']}"


def test_calculate_gap_score_edge_cases():
    """Test gap score calculation handles edge cases properly."""
    config = LLMConfig(
        azure_openai_key="test",
        azure_openai_endpoint="https://test.api.com",
        azure_openai_deployment_name="test-deployment",
        azure_openai_model_name="gpt-4"
    )
    service = EmbeddingService(config)
    
    # Test zero vectors
    zero_u_vector = Vector(values=[0.0, 0.0, 0.0], model="test", dimension=3)
    zero_c_vector = Vector(values=[0.0, 0.0, 0.0], model="test", dimension=3, metadata={"canonical_definition": "test"})
    
    gap_analysis = service.calculate_gap_score(zero_u_vector, zero_c_vector)
    assert gap_analysis["similarity"] == 0.0
    assert gap_analysis["gap_score"] == 1.0
    assert gap_analysis["severity"] == "critical"
    assert gap_analysis["u_vector_magnitude"] == 0.0
    assert gap_analysis["c_vector_magnitude"] == 0.0
    
    # Test dimension mismatch
    u_vector_3d = Vector(values=[1.0, 0.0, 0.0], model="test", dimension=3)
    c_vector_5d = Vector(values=[1.0, 0.0, 0.0, 0.0, 0.0], model="test", dimension=5, metadata={"canonical_definition": "test"})
    
    gap_analysis = service.calculate_gap_score(u_vector_3d, c_vector_5d)
    assert gap_analysis["similarity"] == 0.0  # Should return 0 for dimension mismatch
    assert gap_analysis["gap_score"] == 1.0
    assert gap_analysis["severity"] == "critical"
    
    # Test empty vectors
    empty_u_vector = Vector(values=[], model="test", dimension=0)
    empty_c_vector = Vector(values=[], model="test", dimension=0, metadata={"canonical_definition": "test"})
    
    gap_analysis = service.calculate_gap_score(empty_u_vector, empty_c_vector)
    assert gap_analysis["similarity"] == 0.0
    assert gap_analysis["gap_score"] == 1.0
    assert gap_analysis["severity"] == "critical"
    
    # Test one zero vector, one normal vector
    normal_vector = Vector(values=[1.0, 0.0, 0.0], model="test", dimension=3)
    zero_vector = Vector(values=[0.0, 0.0, 0.0], model="test", dimension=3, metadata={"canonical_definition": "test"})
    
    gap_analysis = service.calculate_gap_score(normal_vector, zero_vector)
    assert gap_analysis["similarity"] == 0.0
    assert gap_analysis["gap_score"] == 1.0
    assert gap_analysis["severity"] == "critical"


def test_gap_score_edge_cases_emphasis_difference():
    """Test gap score edge cases for emphasis difference calculation."""
    config = LLMConfig(
        azure_openai_key="test",
        azure_openai_endpoint="https://test.api.com",
        azure_openai_deployment_name="test-deployment",
        azure_openai_model_name="gpt-4"
    )
    service = EmbeddingService(config)
    
    # Test when c_vector magnitude is zero (should not cause division by zero)
    u_vector = Vector(values=[1.0, 0.0, 0.0], model="test", dimension=3)
    zero_c_vector = Vector(values=[0.0, 0.0, 0.0], model="test", dimension=3, metadata={"canonical_definition": "test"})
    
    gap_analysis = service.calculate_gap_score(u_vector, zero_c_vector)
    
    # Should handle division by zero gracefully
    assert isinstance(gap_analysis["emphasis_difference"], float)
    assert not np.isnan(gap_analysis["emphasis_difference"])
    assert not np.isinf(gap_analysis["emphasis_difference"])
    
    # Test very small c_vector magnitude
    tiny_c_vector = Vector(values=[1e-10, 0.0, 0.0], model="test", dimension=3, metadata={"canonical_definition": "test"})
    
    gap_analysis = service.calculate_gap_score(u_vector, tiny_c_vector)
    
    # Should handle very small denominators
    assert isinstance(gap_analysis["emphasis_difference"], float)
    assert not np.isnan(gap_analysis["emphasis_difference"])
    assert not np.isinf(gap_analysis["emphasis_difference"])