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
                "model": self.llm_config.primary_model,
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

        if self.session_context:
            prompt_parts.append(f"""
## Current Session Context:
- Topic: {self.session_context.get('topic', 'General exploration')}
""")

        # Add persona adjustments from MA
        if self.current_persona_adjustments:
            adjustments_text = "\n".join(self.current_persona_adjustments.get("instructions", []))
            prompt_parts.append(f"""
## Metacognitive Adjustments:
Follow these instructions for the next turn:
- {adjustments_text}
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
    
    async def update_persona(self, adjustments: Optional[Dict[str, Any]]) -> None:
        """Update persona using intelligent LLM-based synthesis with comprehensive validation."""
        if adjustments is None or not adjustments:
            logger.debug("No persona adjustments provided")
            return
        
        # Validate adjustments before processing
        validated_adjustments = self._validate_persona_adjustments(adjustments)
        if not validated_adjustments:
            logger.warning("All persona adjustments failed validation, skipping update")
            return
        
        logger.info(f"Synthesizing validated persona adjustments: {list(validated_adjustments.keys())}")
        
        # Create backup of current state for recovery
        backup_adjustments = self.current_persona_adjustments.copy()
        
        try:
            # Use LLM to intelligently synthesize adjustments with current persona
            synthesized_adjustments = await self._synthesize_persona_adjustments(validated_adjustments)
            
            # Validate synthesized adjustments
            final_adjustments = self._validate_persona_adjustments(synthesized_adjustments)
            if not final_adjustments:
                logger.warning("Synthesized adjustments failed validation, keeping current persona")
                return
            
            # Track previous state for logging changes
            previous_adjustments = self.current_persona_adjustments.copy()
            
            # Apply synthesized adjustments
            self.current_persona_adjustments.update(final_adjustments)
            
            # Validate final persona state
            if not self._validate_persona_state():
                logger.error("Final persona state is invalid, restoring backup")
                self.current_persona_adjustments = backup_adjustments
                return
            
            # Log specific changes made
            for key, value in final_adjustments.items():
                if key in previous_adjustments:
                    if previous_adjustments[key] != value:
                        logger.info(f"Synthesized persona adjustment '{key}': '{previous_adjustments[key]}' -> '{value}'")
                    else:
                        logger.debug(f"Persona adjustment '{key}' unchanged: '{value}'")
                else:
                    logger.info(f"Added synthesized persona adjustment '{key}': '{value}'")
            
            logger.info(f"CA now has {len(self.current_persona_adjustments)} active persona adjustments")
            logger.debug(f"Synthesized persona adjustments: {json.dumps(self.current_persona_adjustments, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error during persona update: {e}")
            logger.info("Restoring persona backup due to error")
            self.current_persona_adjustments = backup_adjustments
    
    def _validate_persona_adjustments(self, adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate persona adjustments to prevent corruption."""
        if not isinstance(adjustments, dict):
            logger.warning(f"Invalid adjustments type: {type(adjustments)}, expected dict")
            return {}
        
        validated = {}
        valid_keys = {
            'metaphor_style', 'question_depth', 'topic_focus', 'pace', 'engagement_style',
            'instructions', 'tone', 'approach', 'guidance', 'behavior', 'style',
            'metaphor_diversity', 'curiosity_level', 'exploration_depth'
        }
        
        for key, value in adjustments.items():
            try:
                # Validate key format
                if not isinstance(key, str):
                    logger.warning(f"Invalid adjustment key type: {type(key)}, skipping")
                    continue
                
                if len(key) > 100:
                    logger.warning(f"Adjustment key too long: {len(key)} chars, skipping")
                    continue
                
                # Validate value
                if value is None:
                    logger.debug(f"Skipping None value for key '{key}'")
                    continue
                
                if isinstance(value, str):
                    if len(value) > 1000:
                        logger.warning(f"Adjustment value too long for '{key}': {len(value)} chars, truncating")
                        value = value[:1000] + "..."
                    
                    # Check for potentially harmful content
                    if any(harmful in value.lower() for harmful in ['ignore', 'forget', 'override', 'disable']):
                        logger.warning(f"Potentially harmful adjustment for '{key}': {value[:50]}..., skipping")
                        continue
                
                elif isinstance(value, list):
                    if len(value) > 20:
                        logger.warning(f"Adjustment list too long for '{key}': {len(value)} items, truncating")
                        value = value[:20]
                    
                    # Validate list items
                    validated_list = []
                    for item in value:
                        if isinstance(item, str) and len(item) <= 200:
                            validated_list.append(item)
                        else:
                            logger.debug(f"Skipping invalid list item in '{key}': {type(item)}")
                    value = validated_list
                
                elif isinstance(value, (int, float, bool)):
                    # Numeric and boolean values are generally safe
                    pass
                
                else:
                    logger.warning(f"Unsupported adjustment value type for '{key}': {type(value)}, skipping")
                    continue
                
                # Add to validated adjustments
                validated[key] = value
                
            except Exception as e:
                logger.error(f"Error validating adjustment '{key}': {e}")
                continue
        
        logger.info(f"Validated {len(validated)}/{len(adjustments)} persona adjustments")
        return validated
    
    def _validate_persona_state(self) -> bool:
        """Validate the overall persona state for consistency."""
        try:
            # Check total size
            total_adjustments = len(self.current_persona_adjustments)
            if total_adjustments > 50:
                logger.warning(f"Too many persona adjustments: {total_adjustments}")
                return False
            
            # Check for conflicting adjustments
            if 'metaphor_style' in self.current_persona_adjustments:
                metaphor_style = self.current_persona_adjustments['metaphor_style']
                if isinstance(metaphor_style, str):
                    if 'avoid metaphors' in metaphor_style.lower() and 'use metaphors' in metaphor_style.lower():
                        logger.warning("Conflicting metaphor style instructions detected")
                        return False
            
            # Check for contradictory pace instructions
            if 'pace' in self.current_persona_adjustments:
                pace = self.current_persona_adjustments['pace']
                if isinstance(pace, str):
                    if 'slow down' in pace.lower() and 'speed up' in pace.lower():
                        logger.warning("Conflicting pace instructions detected")
                        return False
            
            # Validate instructions format
            if 'instructions' in self.current_persona_adjustments:
                instructions = self.current_persona_adjustments['instructions']
                if not isinstance(instructions, list):
                    logger.warning(f"Instructions should be a list, got {type(instructions)}")
                    return False
                
                if len(instructions) > 10:
                    logger.warning(f"Too many instructions: {len(instructions)}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating persona state: {e}")
            return False
    
    async def _synthesize_persona_adjustments(self, new_adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to intelligently synthesize persona adjustments."""
        
        # Get current session context
        session_context = ""
        if hasattr(self, 'session_context') and self.session_context:
            session_context = f"Topic: {self.session_context.get('topic', 'Unknown')}"
        
        # Get recent conversation for context
        recent_conversation = ""
        if self.conversation_history:
            recent_exchanges = self.conversation_history[-6:]  # Last 3 exchanges
            recent_conversation = "\n".join([
                f"{msg['role']}: {msg['content'][:200]}..." if len(msg['content']) > 200 else f"{msg['role']}: {msg['content']}"
                for msg in recent_exchanges
            ])
        
        prompt = f"""
You are helping synthesize persona adjustments for a Socratic learning guide. The goal is to intelligently integrate new behavioral adjustments with existing ones, resolving conflicts and creating coherent guidance.

CURRENT PERSONA ADJUSTMENTS:
{json.dumps(self.current_persona_adjustments, indent=2) if self.current_persona_adjustments else "None"}

NEW ADJUSTMENTS TO INTEGRATE:
{json.dumps(new_adjustments, indent=2)}

SESSION CONTEXT:
{session_context}

RECENT CONVERSATION:
{recent_conversation}

Your task:
1. Intelligently merge new adjustments with existing ones
2. Resolve any conflicts between adjustments
3. Ensure adjustments work together coherently
4. Adapt adjustments to the current conversation context
5. Maintain the Socratic learning approach

Return a JSON object with synthesized adjustments that:
- Preserves effective existing adjustments
- Integrates new adjustments thoughtfully
- Resolves conflicts with context-aware decisions
- Creates coherent behavioral guidance

Focus on these key areas:
- metaphor_style: How to use metaphors effectively
- question_depth: Appropriate question complexity
- topic_focus: How to guide topic boundaries
- pace: Exploration pacing
- engagement_style: How to maintain curiosity

Example format:
{{
    "metaphor_style": "Encourage diverse metaphors from nature and technology domains",
    "question_depth": "Use deeper probing questions to challenge assumptions",
    "topic_focus": "Gently redirect when conversation drifts from core concept",
    "pace": "Slow down to allow deeper reflection",
    "engagement_style": "Use more encouraging language to build confidence"
}}
"""

        try:
            response = await self._make_completion_request(
                [{"role": "user", "content": prompt}],
                temperature=0.4,  # Balanced creativity and consistency
                max_tokens=300
            )
            
            # Parse JSON response
            synthesized_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', synthesized_text, re.DOTALL)
            if json_match:
                synthesized_adjustments = json.loads(json_match.group())
                logger.info("Successfully synthesized persona adjustments using LLM")
                return synthesized_adjustments
            else:
                logger.warning("Could not parse JSON from LLM response, using direct merge")
                return new_adjustments
                
        except Exception as e:
            logger.error(f"Failed to synthesize persona adjustments: {e}")
            # Fallback to simple merge
            return new_adjustments
    
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