"""Tests for the Orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from peengine.core.orchestrator import ExplorationEngine
from peengine.models.config import Settings
from peengine.models.graph import Session, Vector

# A mock settings object for initialization
@pytest.fixture
def mock_settings():
    """Provides a mock Settings object."""
    with patch('peengine.models.config.Settings') as MockSettings:
        instance = MockSettings.return_value
        instance.database_config = MagicMock()
        instance.mongodb_uri = "mongodb://localhost:27017"
        instance.mongodb_database = "test_db"
        instance.llm_config = MagicMock()
        instance.persona_config = MagicMock()
        yield instance

@pytest.fixture
def orchestrator(mock_settings):
    """Provides an instance of the ExplorationEngine with mocked components."""
    with patch('peengine.core.orchestrator.Neo4jClient'), \
         patch('peengine.core.orchestrator.MongoDBClient'), \
         patch('peengine.core.orchestrator.ConversationalAgent') as MockCA, \
         patch('peengine.core.orchestrator.PatternDetector'), \
         patch('peengine.core.orchestrator.MetacognitiveAgent') as MockMA, \
         patch('peengine.core.orchestrator.EmbeddingService') as MockEmbeddingService, \
         patch('peengine.core.orchestrator.AnalyticsEngine'):
        
        engine = ExplorationEngine(mock_settings)
        engine.ca = MockCA()
        engine.ma = MockMA()
        engine.embedding_service = MockEmbeddingService()
        engine.current_session = Session(id="test_session", title="Test Session", topic="testing")
        engine.current_session.messages = [{"user": "...", "assistant": "..."}]
        return engine

@pytest.fixture
def sample_session():
    """Provides a sample session with messages."""
    session = Session(id="test_session", title="Quantum Mechanics Session", topic="quantum mechanics")
    session.messages = [
        {
            "timestamp": datetime.now().isoformat(),
            "user": "What is quantum superposition?",
            "assistant": "Think of it like a coin spinning in the air..."
        },
        {
            "timestamp": datetime.now().isoformat(),
            "user": "How does it relate to uncertainty?",
            "assistant": "The spinning coin metaphor helps us understand..."
        }
    ]
    return session

@pytest.fixture
def sample_vectors():
    """Provides sample vectors for testing."""
    u_vector = Vector(
        values=[0.1, 0.2, 0.3, 0.4, 0.5],
        model="text-embedding-ada-002",
        dimension=5
    )
    c_vector = Vector(
        values=[0.15, 0.25, 0.35, 0.45, 0.55],
        model="text-embedding-ada-002", 
        dimension=5,
        metadata={"canonical_definition": "A quantum mechanical principle..."}
    )
    return u_vector, c_vector

# Existing tests
@pytest.mark.asyncio
async def test_gap_check_happy_path(orchestrator):
    """Test the successful execution of the _gap_check method."""
    # Arrange
    orchestrator._identify_recent_concept = AsyncMock(return_value="test_concept")
    orchestrator._get_user_vector = AsyncMock(return_value=Vector(values=[1.0]))
    orchestrator._get_or_create_canonical_vector = AsyncMock(return_value=Vector(values=[0.9]))
    orchestrator.embedding_service.calculate_gap_score = MagicMock(return_value={
        "similarity": 0.95, "gap_score": 0.05, "severity": "low", "canonical_definition": "..."
    })
    orchestrator._format_gap_message = AsyncMock(return_value="A friendly message.")

    # Act
    result = await orchestrator._gap_check()

    # Assert
    orchestrator._identify_recent_concept.assert_called_once()
    orchestrator._get_user_vector.assert_called_once_with("test_concept")
    orchestrator._get_or_create_canonical_vector.assert_called_once_with("test_concept")
    orchestrator.embedding_service.calculate_gap_score.assert_called_once()
    orchestrator._format_gap_message.assert_called_once()
    assert result["concept"] == "test_concept"
    assert result["message"] == "A friendly message."
    assert "error" not in result

@pytest.mark.asyncio
async def test_gap_check_no_concept(orchestrator):
    """Test _gap_check when no recent concept can be identified."""
    # Arrange
    orchestrator._identify_recent_concept = AsyncMock(return_value=None)

    # Act
    result = await orchestrator._gap_check()

    # Assert
    assert "error" in result
    assert "No clear concept" in result["error"]

@pytest.mark.asyncio
async def test_gap_check_no_u_vector(orchestrator):
    """Test _gap_check when the user vector cannot be found."""
    # Arrange
    orchestrator._identify_recent_concept = AsyncMock(return_value="test_concept")
    orchestrator._get_user_vector = AsyncMock(return_value=None)

    # Act
    result = await orchestrator._gap_check()

    # Assert
    assert "error" in result
    assert "No user understanding vector" in result["error"]

@pytest.mark.asyncio
async def test_gap_check_no_c_vector(orchestrator):
    """Test _gap_check when the canonical vector cannot be created."""
    # Arrange
    orchestrator._identify_recent_concept = AsyncMock(return_value="test_concept")
    orchestrator._get_user_vector = AsyncMock(return_value=Vector(values=[1.0]))
    orchestrator._get_or_create_canonical_vector = AsyncMock(return_value=None)

    # Act
    result = await orchestrator._gap_check()

    # Assert
    assert "error" in result
    assert "Failed to obtain canonical vector" in result["error"]

# New comprehensive tests for task 6

@pytest.mark.asyncio
async def test_gap_check_with_existing_vectors(orchestrator, sample_vectors):
    """Test gap analysis when both u_vector and c_vector exist."""
    u_vector, c_vector = sample_vectors
    
    # Arrange
    orchestrator._identify_recent_concept = AsyncMock(return_value="quantum_superposition")
    orchestrator._get_user_vector = AsyncMock(return_value=u_vector)
    orchestrator._get_or_create_canonical_vector = AsyncMock(return_value=c_vector)
    
    gap_analysis = {
        "similarity": 0.92,
        "gap_score": 0.08,
        "severity": "minimal",
        "canonical_definition": "A quantum mechanical principle...",
        "severity_description": "Very close alignment"
    }
    orchestrator.embedding_service.calculate_gap_score = MagicMock(return_value=gap_analysis)
    orchestrator._format_gap_message = AsyncMock(return_value="Great understanding! Your metaphors align well...")

    # Act
    result = await orchestrator._gap_check()

    # Assert
    assert result["concept"] == "quantum_superposition"
    assert result["similarity"] == 0.92
    assert result["gap_score"] == 0.08
    assert result["severity"] == "minimal"
    assert result["message"] == "Great understanding! Your metaphors align well..."
    assert "error" not in result
    
    # Verify method calls
    orchestrator._identify_recent_concept.assert_called_once()
    orchestrator._get_user_vector.assert_called_once_with("quantum_superposition")
    orchestrator._get_or_create_canonical_vector.assert_called_once_with("quantum_superposition")
    orchestrator.embedding_service.calculate_gap_score.assert_called_once_with(u_vector, c_vector)
    orchestrator._format_gap_message.assert_called_once_with("quantum_superposition", gap_analysis)

@pytest.mark.asyncio
async def test_gap_check_creates_missing_c_vector(orchestrator, sample_vectors):
    """Test gap check creates c_vector when missing."""
    u_vector, c_vector = sample_vectors
    
    # Arrange
    orchestrator._identify_recent_concept = AsyncMock(return_value="new_concept")
    orchestrator._get_user_vector = AsyncMock(return_value=u_vector)
    
    # Mock that c_vector doesn't exist initially, then gets created
    orchestrator._get_or_create_canonical_vector = AsyncMock(return_value=c_vector)
    
    gap_analysis = {
        "similarity": 0.75,
        "gap_score": 0.25,
        "severity": "moderate",
        "canonical_definition": "A newly created canonical definition...",
        "severity_description": "Noticeable differences"
    }
    orchestrator.embedding_service.calculate_gap_score = MagicMock(return_value=gap_analysis)
    orchestrator._format_gap_message = MagicMock(return_value="Interesting perspective! Let's explore...")

    # Act
    result = await orchestrator._gap_check()

    # Assert
    assert result["concept"] == "new_concept"
    assert result["similarity"] == 0.75
    assert result["severity"] == "moderate"
    assert "error" not in result
    
    # Verify that _get_or_create_canonical_vector was called (which handles creation)
    orchestrator._get_or_create_canonical_vector.assert_called_once_with("new_concept")

@pytest.mark.asyncio
async def test_persona_adjustment_applied(orchestrator):
    """Test that MA persona adjustments are applied to CA."""
    # Arrange
    user_input = "I keep using the same water metaphor for everything"
    
    # Mock CA response
    ca_response = {
        "message": "Let's explore that metaphor further...",
        "reasoning": {"metaphor_used": "water flow"}
    }
    orchestrator.ca.process_input = AsyncMock(return_value=ca_response)
    
    # Mock PD extractions
    extractions = []
    orchestrator.pd.extract_patterns = AsyncMock(return_value=extractions)
    
    # Mock MA analysis with persona adjustments
    ma_analysis = {
        "insights": ["User is locked into water metaphors"],
        "flags": ["metaphor_lock"],
        "persona_adjustments": {
            "metaphor_diversity": "encourage_new_domains",
            "prompting_style": "suggest_alternative_metaphors"
        }
    }
    orchestrator.ma.analyze_session = AsyncMock(return_value=ma_analysis)
    
    # Mock CA persona update
    orchestrator.ca.update_persona = AsyncMock()
    
    # Mock database operations
    orchestrator.message_db.add_message_exchange = AsyncMock()
    orchestrator.message_db.update_session_analysis = AsyncMock()

    # Act
    result = await orchestrator.process_user_input(user_input)

    # Assert
    # Verify CA.update_persona was called with the adjustments
    orchestrator.ca.update_persona.assert_called_once_with({
        "metaphor_diversity": "encourage_new_domains",
        "prompting_style": "suggest_alternative_metaphors"
    })
    
    # Verify the result contains expected data
    assert result["message"] == ca_response["message"]
    assert result["ma_insights"] == ["User is locked into water metaphors"]

@pytest.mark.asyncio
async def test_identify_recent_concept(orchestrator, sample_session):
    """Test concept extraction from conversation."""
    # Arrange
    orchestrator.current_session = sample_session
    
    # Mock the LLM response for concept identification
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "quantum superposition"
    
    orchestrator.ca.client.chat.completions.create = AsyncMock(return_value=mock_response)
    orchestrator.ca._get_deployment_name = MagicMock(return_value="gpt-4")

    # Act
    result = await orchestrator._identify_recent_concept()

    # Assert
    assert result == "quantum superposition"
    
    # Verify LLM was called with appropriate prompt
    orchestrator.ca.client.chat.completions.create.assert_called_once()
    call_args = orchestrator.ca.client.chat.completions.create.call_args
    
    # Check that the prompt contains the conversation history
    prompt_content = call_args[1]["messages"][0]["content"]
    assert "What is quantum superposition?" in prompt_content
    assert "How does it relate to uncertainty?" in prompt_content

@pytest.mark.asyncio
async def test_identify_recent_concept_no_session(orchestrator):
    """Test concept identification when no session exists."""
    # Arrange
    orchestrator.current_session = None

    # Act
    result = await orchestrator._identify_recent_concept()

    # Assert
    assert result is None

@pytest.mark.asyncio
async def test_identify_recent_concept_empty_messages(orchestrator):
    """Test concept identification when session has no messages."""
    # Arrange
    orchestrator.current_session = Session(id="empty_session", title="Empty Session", topic="test")
    orchestrator.current_session.messages = []

    # Act
    result = await orchestrator._identify_recent_concept()

    # Assert
    assert result is None

@pytest.mark.asyncio
async def test_identify_recent_concept_llm_failure(orchestrator, sample_session):
    """Test concept identification fallback when LLM fails."""
    # Arrange
    orchestrator.current_session = sample_session
    
    # Mock LLM failure
    orchestrator.ca.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
    orchestrator.ca._get_deployment_name = MagicMock(return_value="gpt-4")

    # Act
    result = await orchestrator._identify_recent_concept()

    # Assert - should fallback to session topic
    assert result == "quantum mechanics"

@pytest.mark.asyncio
async def test_format_gap_message_high_similarity(orchestrator):
    """Test gap message formatting for high similarity."""
    # Arrange
    concept = "quantum_tunneling"
    gap_analysis = {
        "similarity": 0.95,
        "gap_score": 0.05,
        "severity": "minimal",
        "canonical_definition": "A quantum mechanical phenomenon...",
        "severity_description": "Very close alignment"
    }
    
    # Mock the LLM-powered message generation to fail, testing fallback
    orchestrator.ca.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
    orchestrator.ca._get_deployment_name = MagicMock(return_value="gpt-4")
    orchestrator._extract_user_metaphors = AsyncMock(return_value="No recent metaphors detected")

    # Act
    result = await orchestrator._format_gap_message(concept, gap_analysis)

    # Assert - should use fallback message
    assert "quantum_tunneling" in result
    assert "95% alignment" in result
    assert "excellent work" in result.lower()

@pytest.mark.asyncio
async def test_format_gap_message_low_similarity(orchestrator):
    """Test gap message formatting for low similarity."""
    # Arrange
    concept = "wave_function"
    gap_analysis = {
        "similarity": 0.45,
        "gap_score": 0.55,
        "severity": "high",
        "canonical_definition": "A mathematical description...",
        "severity_description": "Significant gaps"
    }
    
    # Mock the LLM-powered message generation to fail, testing fallback
    orchestrator.ca.client.chat.completions.create = AsyncMock(side_effect=Exception("API Error"))
    orchestrator.ca._get_deployment_name = MagicMock(return_value="gpt-4")
    orchestrator._extract_user_metaphors = AsyncMock(return_value="No recent metaphors detected")

    # Act
    result = await orchestrator._format_gap_message(concept, gap_analysis)

    # Assert - should use fallback message
    assert "wave_function" in result
    assert "45% alignment" in result
    assert "unique angle" in result.lower()

@pytest.mark.asyncio
async def test_format_gap_message_with_llm_success(orchestrator):
    """Test gap message formatting when LLM generation succeeds."""
    # Arrange
    concept = "entanglement"
    gap_analysis = {
        "similarity": 0.80,
        "gap_score": 0.20,
        "severity": "low",
        "canonical_definition": "A quantum phenomenon where particles...",
        "severity_description": "Good alignment"
    }
    
    # Mock successful LLM response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "🎯 Your understanding of entanglement shows great intuition! Your metaphors capture the essence..."
    
    orchestrator.ca.client.chat.completions.create = AsyncMock(return_value=mock_response)
    orchestrator.ca._get_deployment_name = MagicMock(return_value="gpt-4")
    orchestrator._extract_user_metaphors = AsyncMock(return_value="spooky action, invisible threads")
    
    # Mock session for context
    orchestrator.current_session = Session(id="test", title="Test Session", topic="quantum physics")
    orchestrator.current_session.messages = [{"user": "test", "assistant": "test"}]

    # Act
    result = await orchestrator._format_gap_message(concept, gap_analysis)

    # Assert
    assert result == "🎯 Your understanding of entanglement shows great intuition! Your metaphors capture the essence..."
    
    # Verify LLM was called
    orchestrator.ca.client.chat.completions.create.assert_called_once()

@pytest.mark.asyncio
async def test_persona_adjustment_not_applied_when_none(orchestrator):
    """Test that persona adjustment is not applied when MA provides no adjustments."""
    # Arrange
    user_input = "Tell me about photons"
    
    # Mock CA response
    ca_response = {
        "message": "Photons are like packets of light energy...",
        "reasoning": {"metaphor_used": "energy packets"}
    }
    orchestrator.ca.process_input = AsyncMock(return_value=ca_response)
    
    # Mock PD extractions
    extractions = []
    orchestrator.pd.extract_patterns = AsyncMock(return_value=extractions)
    
    # Mock MA analysis WITHOUT persona adjustments
    ma_analysis = {
        "insights": ["User is exploring light concepts"],
        "flags": [],
        # No persona_adjustments key
    }
    orchestrator.ma.analyze_session = AsyncMock(return_value=ma_analysis)
    
    # Mock CA persona update
    orchestrator.ca.update_persona = AsyncMock()
    
    # Mock database operations
    orchestrator.message_db.add_message_exchange = AsyncMock()
    orchestrator.message_db.update_session_analysis = AsyncMock()

    # Act
    result = await orchestrator.process_user_input(user_input)

    # Assert
    # Verify CA.update_persona was NOT called
    orchestrator.ca.update_persona.assert_not_called()
    
    # Verify the result contains expected data
    assert result["message"] == ca_response["message"]
    assert result["ma_insights"] == ["User is exploring light concepts"]
