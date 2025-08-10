"""Tests for the ConversationalAgent."""

import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from peengine.agents.conversational import ConversationalAgent
from peengine.models.config import LLMConfig, PersonaConfig

# Fixture for the ConversationalAgent
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
def persona_config():
    """Provides a mock PersonaConfig."""
    return PersonaConfig(path="/fake/path/persona.md")

@pytest.fixture
def conversational_agent(llm_config, persona_config):
    """Provides an instance of the ConversationalAgent with a mocked client."""
    agent = ConversationalAgent(llm_config, persona_config)
    agent.client = AsyncMock()
    return agent





@pytest.mark.asyncio
async def test_update_persona_appends_instructions(conversational_agent):
    """Test that update_persona appends to an existing list of instructions."""
    # Arrange
    conversational_agent.current_persona_adjustments = {"instructions": ["Be concise."]}
    adjustments = {"instruction": "Use more metaphors."}

    # Act
    await conversational_agent.update_persona(adjustments)

    # Assert
    assert len(conversational_agent.current_persona_adjustments["instructions"]) == 2
    assert conversational_agent.current_persona_adjustments["instructions"][1] == "Use more metaphors."

@pytest.mark.asyncio
async def test_build_system_prompt_with_adjustments(conversational_agent):
    """Test that the system prompt correctly includes persona adjustments."""
    # Arrange
    conversational_agent.base_persona = "Base persona."
    conversational_agent.current_persona_adjustments = {"instructions": ["Instruction 1.", "Instruction 2."]}
    conversational_agent.session_context = {"topic": "test_topic"}

    # Act
    prompt = conversational_agent._build_system_prompt()

    # Assert
    assert "Base persona." in prompt
    assert "Metacognitive Adjustments" in prompt
    assert "- Instruction 1." in prompt
    assert "- Instruction 2." in prompt
