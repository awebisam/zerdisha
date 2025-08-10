"""Tests for the MetacognitiveAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from peengine.agents.metacognitive import MetacognitiveAgent
from peengine.models.config import LLMConfig
from peengine.models.graph import Session

# Fixture for the MetacognitiveAgent
@pytest.fixture
def llm_config():
    """Provides a mock LLMConfig."""
    return LLMConfig(
        azure_openai_key="test_key",
        azure_openai_endpoint="test_endpoint",
        azure_openai_api_version="test_version",
        azure_openai_deployment_name="test_deployment",
        azure_openai_model_name="gpt-4"
    )

@pytest.fixture
def metacognitive_agent(llm_config):
    """Provides an instance of the MetacognitiveAgent with a mocked client."""
    agent = MetacognitiveAgent(llm_config)
    agent.client = AsyncMock()
    return agent

@pytest.mark.asyncio
async def test_analyze_session_success(metacognitive_agent):
    """Test the happy path for analyze_session with a valid LLM response."""
    # Arrange
    mock_session = Session(id="test_session", title="testing", topic="testing")
    mock_session.messages = [{"user": "hello", "assistant": "world"}]

    metacognitive_agent._analyze_metaphor_patterns = AsyncMock(return_value={})

    expected_analysis = {
        "flags": [{"type": "metaphor_lock", "severity": "high", "evidence": "..."}],
        "persona_adjustments": {"instruction": "Change metaphor domain."}
    }
    
    # Mock the LLM response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(expected_analysis)
    metacognitive_agent.client.chat.completions.create.return_value = mock_response

    # Act
    analysis = await metacognitive_agent.analyze_session(mock_session, [], {})

    # Assert
    metacognitive_agent.client.chat.completions.create.assert_called_once()
    assert all(item in analysis.items() for item in expected_analysis.items())
    assert "learning_trajectory" in analysis
    assert "curiosity_health" in analysis

@pytest.mark.asyncio
async def test_analyze_session_llm_error(metacognitive_agent):
    """Test that analyze_session handles an exception from the LLM client gracefully."""
    # Arrange
    mock_session = Session(id="test_session", title="testing", topic="testing")
    metacognitive_agent._analyze_metaphor_patterns = AsyncMock(return_value={})
    metacognitive_agent.client.chat.completions.create.side_effect = Exception("LLM is down")

    # Act
    analysis = await metacognitive_agent.analyze_session(mock_session, [], {})

    # Assert
    assert "error" in analysis

@pytest.mark.asyncio
async def test_analyze_session_invalid_json(metacognitive_agent):
    """Test that analyze_session handles malformed JSON from the LLM."""
    # Arrange
    mock_session = Session(id="test_session", title="testing", topic="testing")
    
    metacognitive_agent._analyze_metaphor_patterns = AsyncMock(return_value={})
    
    # Mock the LLM response with invalid JSON
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "this is not json"
    metacognitive_agent.client.chat.completions.create.return_value = mock_response

    # Act
    analysis = await metacognitive_agent.analyze_session(mock_session, [], {})

    # Assert
    assert "error" in analysis

@pytest.mark.asyncio
async def test_analyze_session_missing_keys(metacognitive_agent):
    """Test that analyze_session validates the structure of the LLM's JSON response."""
    # Arrange
    mock_session = Session(id="test_session", title="testing", topic="testing")
    
    metacognitive_agent._analyze_metaphor_patterns = AsyncMock(return_value={})
    
    # Mock the LLM response with missing keys
    invalid_analysis = {"some_other_key": "some_value"}
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(invalid_analysis)
    metacognitive_agent.client.chat.completions.create.return_value = mock_response

    # Act
    analysis = await metacognitive_agent.analyze_session(mock_session, [], {})

    # Assert
    assert "error" in analysis
