"""Conversational Agent - The Socratic guide using metaphors."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

from openai import AsyncAzureOpenAI

from ..models.config import LLMConfig, PersonaConfig

logger = logging.getLogger(__name__)


class ConversationalAgent:
    """Socratic conversational agent that guides exploration through metaphors."""
    
    def __init__(self, llm_config: LLMConfig, persona_config: PersonaConfig):
        self.llm_config = llm_config
        self.persona_config = persona_config
        
        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=llm_config.azure_openai_key,
            azure_endpoint=llm_config.azure_openai_endpoint,
            api_version=llm_config.azure_openai_api_version
        )
        
        # Fallback client for Azure AI Foundry (if configured)
        self.fallback_client = None
        if llm_config.azure_ai_foundry_key and llm_config.azure_ai_foundry_endpoint:
            self.fallback_client = AsyncAzureOpenAI(
                api_key=llm_config.azure_ai_foundry_key,
                azure_endpoint=llm_config.azure_ai_foundry_endpoint,
                api_version=llm_config.azure_openai_api_version
            )
        
        # Persona and context
        self.base_persona = ""
        self.current_persona_adjustments = {}
        self.session_context = {}
        
        # Conversation history for current session
        self.conversation_history: List[Dict[str, str]] = []
    
    def _get_deployment_name(self) -> str:
        """Get the deployment name for Azure OpenAI."""
        return self.llm_config.azure_openai_deployment_name
    
    async def _make_completion_request(self, messages: List[Dict[str, str]], temperature: float = None, max_tokens: int = None):
        """Make completion request with fallback support."""
        temperature = temperature or self.llm_config.temperature
        max_tokens = max_tokens or self.llm_config.max_tokens
        
        try:
            # Try primary Azure OpenAI client
            response = await self.client.chat.completions.create(
                model=self._get_deployment_name(),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            logger.warning(f"Primary Azure OpenAI failed: {e}")
            
            # Try fallback client if available
            if self.fallback_client:
                try:
                    response = await self.fallback_client.chat.completions.create(
                        model=self.llm_config.fallback_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    logger.info("Used fallback Azure AI Foundry client")
                    return response
                except Exception as fallback_e:
                    logger.error(f"Fallback Azure AI Foundry also failed: {fallback_e}")
            
            # Re-raise original exception if no fallback or fallback failed
            raise e
        
    async def initialize(self) -> None:
        """Initialize the agent by loading persona."""
        await self._load_persona()
        logger.info("Conversational Agent initialized")
    
    async def _load_persona(self) -> None:
        """Load persona from markdown file."""
        try:
            persona_path = Path(self.persona_config.path)
            if persona_path.exists():
                self.base_persona = persona_path.read_text(encoding='utf-8')
                logger.info(f"Loaded persona from {persona_path}")
            else:
                logger.warning(f"Persona file not found: {persona_path}")
                self.base_persona = self._get_default_persona()
        except Exception as e:
            logger.error(f"Failed to load persona: {e}")
            self.base_persona = self._get_default_persona()
    
    def _get_default_persona(self) -> str:
        """Get default persona if file loading fails."""
        return '''
# Socratic Learning Guide

You are a Socratic learning guide focused on exploration through metaphors and questions.

## Core Principles:
- Never give direct answers - always respond with questions that guide discovery
- Use metaphors as the primary communication method
- Adapt metaphors if the learner doesn't understand
- Challenge assumptions gently but persistently  
- Keep the learner exploring rather than settling on premature closure
- Focus on "why" and "how" questions rather than "what"

## Interaction Style:
- Start with the learner's current mental model
- Build on their existing metaphors and analogies
- Switch metaphors if one isn't working
- Guide toward deeper understanding through questioning
- Encourage hypothesis formation and testing

## Avoid:
- Lecture-style explanations
- Immediate answers to questions
- Academic jargon without metaphorical bridges
- Overwhelming with too many concepts at once
'''
    
    async def start_session(self, topic: str, relevant_context: List[Dict[str, Any]]) -> None:
        """Start a new session with topic and relevant context."""
        self.session_context = {
            "topic": topic,
            "relevant_context": relevant_context,
            "session_start": True
        }
        self.conversation_history = []
        
        logger.info(f"CA started session for topic: {topic}")
    
    async def process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate Socratic response."""
        
        # Build conversation context
        system_prompt = self._build_system_prompt()
        
        # Add user input to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Build messages for OpenAI
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history (last 10 exchanges to avoid token limits)
        recent_history = self.conversation_history[-20:]  # Last 10 user-assistant pairs
        messages.extend(recent_history)
        
        try:
            response = await self._make_completion_request(messages)
            
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant", 
                "content": assistant_message
            })
            
            # Extract reasoning/metadata if needed
            reasoning = {
                "model": self.llm_config.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "finish_reason": response.choices[0].finish_reason
            }
            
            return {
                "message": assistant_message,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"Failed to get LLM response: {e}")
            return {
                "message": "I apologize, but I'm having trouble processing that right now. Could you rephrase your thought?",
                "reasoning": {"error": str(e)}
            }
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt with persona and context."""
        prompt_parts = [self.base_persona]
        
        # Add session context
        if self.session_context:
            prompt_parts.append(f"""
## Current Session Context:
- Topic: {self.session_context.get('topic', 'General exploration')}
- Session stage: {'Beginning of exploration' if self.session_context.get('session_start') else 'Ongoing exploration'}
""")
            
            # Add relevant context from past sessions
            if self.session_context.get('relevant_context'):
                context_summary = self._summarize_relevant_context(
                    self.session_context['relevant_context']
                )
                prompt_parts.append(f"""
## Relevant Past Knowledge:
{context_summary}
""")
        
        # Add persona adjustments from MA
        if self.current_persona_adjustments:
            prompt_parts.append(f"""
## Current Adjustments:
{json.dumps(self.current_persona_adjustments, indent=2)}
""")
        
        # Add specific instructions for current conversation stage
        conversation_length = len(self.conversation_history)
        if conversation_length == 0:
            prompt_parts.append("""
## Opening Instructions:
This is the start of the session. Begin by understanding the learner's current mental model of the topic. Ask about their existing understanding or analogies they might already use.
""")
        elif conversation_length < 10:
            prompt_parts.append("""
## Early Stage Instructions:
You're in the exploration phase. Focus on expanding their mental model through metaphors and gentle questioning. Don't settle on simple explanations yet.
""")
        else:
            prompt_parts.append("""
## Deeper Exploration Instructions:
The learner has shared some thoughts. Now deepen the exploration by challenging assumptions, exploring edge cases of their metaphors, or connecting to new domains.
""")
        
        return "\n".join(prompt_parts)
    
    def _summarize_relevant_context(self, context_nodes: List[Dict[str, Any]]) -> str:
        """Summarize relevant context from past sessions."""
        if not context_nodes:
            return "No relevant past explorations found."
        
        summaries = []
        for node in context_nodes[:5]:  # Limit to most relevant
            node_type = node.get('node_type', 'concept')
            label = node.get('label', 'Unknown')
            properties = node.get('properties', {})
            
            if node_type == 'concept':
                domain = properties.get('domain', 'unknown domain')
                summary = f"- Previously explored '{label}' in {domain}"
            elif node_type == 'metaphor':
                concept = properties.get('concept', 'unknown concept')
                summary = f"- Used metaphor '{label}' to understand {concept}"
            else:
                summary = f"- {node_type.title()}: {label}"
            
            summaries.append(summary)
        
        return "\n".join(summaries)
    
    async def update_persona(self, adjustments: Dict[str, Any]) -> None:
        """Update persona based on MA feedback."""
        self.current_persona_adjustments.update(adjustments)
        logger.info(f"Updated persona with adjustments: {list(adjustments.keys())}")
    
    async def get_canonical_check(self, concept: str) -> Dict[str, Any]:
        """Perform canonical knowledge check for a concept."""
        # This would be called by the CA when it needs to verify understanding
        # against canonical sources
        
        system_prompt = f"""
You are checking the canonical/academic understanding of: {concept}

Provide:
1. Standard academic definition
2. Key properties or characteristics  
3. Common misconceptions
4. Related canonical concepts

Be concise and factual.
"""
        
        try:
            response = await self._make_completion_request(
                [{"role": "system", "content": system_prompt}],
                temperature=0.3,  # Lower temperature for factual content
                max_tokens=500
            )
            
            return {
                "concept": concept,
                "canonical_info": response.choices[0].message.content,
                "confidence": "high"  # Could be computed based on response
            }
            
        except Exception as e:
            logger.error(f"Failed canonical check for {concept}: {e}")
            return {
                "concept": concept,
                "canonical_info": "Unable to retrieve canonical information",
                "confidence": "low",
                "error": str(e)
            }
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about the current session."""
        return {
            "total_exchanges": len(self.conversation_history) // 2,  # Pairs of user/assistant
            "topic": self.session_context.get('topic'),
            "persona_adjustments": len(self.current_persona_adjustments),
            "has_context": bool(self.session_context.get('relevant_context'))
        }