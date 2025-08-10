"""Tests for the Orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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
         patch('peengine.core.orchestrator.MetacognitiveAgent'), \
         patch('peengine.core.orchestrator.EmbeddingService') as MockEmbeddingService, \
         patch('peengine.core.orchestrator.AnalyticsEngine'):
        
        engine = ExplorationEngine(mock_settings)
        engine.ca = MockCA()
        engine.embedding_service = MockEmbeddingService()
        engine.current_session = Session(id="test_session", topic="testing")
        engine.current_session.messages = [{"user": "...", "assistant": "..."}]
        return engine

@pytest.mark.asyncio
async def test_gap_check_happy_path(orchestrator):
    """Test the successful execution of the _gap_check method."""
    # Arrange
    orchestrator._identify_recent_concept = AsyncMock(return_value="test_concept")
    orchestrator._get_user_vector = AsyncMock(return_value=Vector(values=[1.0]))
    orchestrator.embedding_service.get_or_create_c_vector = AsyncMock(return_value=Vector(values=[0.9]))
    orchestrator.embedding_service.calculate_gap_score = MagicMock(return_value={
        "similarity": 0.95, "gap_score": 0.05, "severity": "low", "canonical_definition": "..."
    })
    orchestrator.ca.generate_gap_analysis_message = AsyncMock(return_value="A friendly message.")

    # Act
    result = await orchestrator._gap_check()

    # Assert
    orchestrator._identify_recent_concept.assert_called_once()
    orchestrator._get_user_vector.assert_called_once_with("test_concept")
    orchestrator.embedding_service.get_or_create_c_vector.assert_called_once()
    orchestrator.embedding_service.calculate_gap_score.assert_called_once()
    orchestrator.ca.generate_gap_analysis_message.assert_called_once()
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
    orchestrator.embedding_service.get_or_create_c_vector = AsyncMock(return_value=None)

    # Act
    result = await orchestrator._gap_check()

    # Assert
    assert "error" in result
    assert "Failed to obtain canonical vector" in result["error"]
