"""Pattern Detector - Extracts concepts and relationships from conversations."""

import logging
import json
import re
from typing import Dict, List, Optional, Any

from openai import AsyncOpenAI, AsyncAzureOpenAI

from ..models.config import LLMConfig
from ..models.graph import ConceptExtraction, MetaphorsConnection

logger = logging.getLogger(__name__)


class PatternDetector:
    """Extracts conceptual patterns and relationships from conversations."""
    
    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        
        # Initialize the appropriate client based on API type
        if llm_config.api_type == "azure":
            self.client = AsyncAzureOpenAI(
                api_key=llm_config.api_key,
                azure_endpoint=llm_config.base_url,
                api_version=llm_config.api_version
            )
        else:
            self.client = AsyncOpenAI(
                api_key=llm_config.api_key,
                base_url=llm_config.base_url
            )
        
        # Pattern extraction templates
        self.extraction_templates = self._load_extraction_templates()
    
    def _get_model_name(self) -> str:
        """Get the appropriate model name or deployment for API calls."""
        return (
            self.llm_config.deployment_name 
            if self.llm_config.api_type == "azure" and self.llm_config.deployment_name
            else self.llm_config.model
        )
    
    async def initialize(self) -> None:
        """Initialize the pattern detector."""
        logger.info("Pattern Detector initialized")
    
    def _load_extraction_templates(self) -> Dict[str, str]:
        """Load templates for different types of pattern extraction."""
        return {
            "concept_extraction": '''
Extract conceptual information from this conversation exchange.

USER INPUT: {user_input}
ASSISTANT RESPONSE: {assistant_response}

Extract and return JSON with the following structure:
{{
    "concepts": [
        {{
            "concept": "concept name",
            "domain": "domain/field",
            "metaphors": ["metaphor1", "metaphor2"],
            "confidence": 0.8,
            "context": "brief context where mentioned",
            "relationships": [
                {{
                    "type": "builds_on|contradicts|explores|similar_to",
                    "target": "other concept",
                    "description": "how they relate"
                }}
            ]
        }}
    ],
    "metaphor_connections": [
        {{
            "source_concept": "concept A",
            "target_concept": "concept B", 
            "metaphor": "the metaphor linking them",
            "domains": ["domain1", "domain2"],
            "strength": 0.7,
            "bidirectional": true
        }}
    ],
    "domain_info": {{
        "primary_domain": "main field being explored",
        "connected_domains": ["other connected fields"]
    }}
}}

Focus on:
- New concepts introduced or explored
- Metaphors and analogies used
- Cross-domain connections
- Relationships between ideas
- Level of understanding depth

Return only valid JSON.
''',
            "metaphor_analysis": '''
Analyze the metaphorical content in this exchange:

USER: {user_input}
ASSISTANT: {assistant_response}

Identify:
1. All metaphors and analogies used
2. How well they map to the target concept
3. What aspects they illuminate vs obscure
4. Cross-domain bridges created

Return JSON:
{{
    "metaphors": [
        {{
            "metaphor": "the metaphor text",
            "source_domain": "where metaphor comes from",
            "target_concept": "what it explains",
            "effectiveness": 0.8,
            "limitations": "what it doesn't capture",
            "extensions": ["how it could be extended"]
        }}
    ],
    "cross_domain_links": [
        {{
            "from_domain": "source field",
            "to_domain": "target field", 
            "bridge_concept": "connecting idea",
            "strength": 0.7
        }}
    ]
}}
''',
            "learning_progression": '''
Analyze the learning progression in this conversation:

CONVERSATION CONTEXT: {context}
LATEST EXCHANGE:
USER: {user_input}
ASSISTANT: {assistant_response}

Assess:
1. Depth of understanding shown
2. Conceptual breakthroughs or "aha" moments  
3. Persistent confusions or blocks
4. Readiness for deeper/canonical knowledge

Return JSON:
{{
    "understanding_depth": {{
        "level": "surface|developing|deep|expert",
        "evidence": "what shows this level"
    }},
    "breakthroughs": [
        {{
            "concept": "what was understood",
            "indicator": "how we know they got it"
        }}
    ],
    "confusions": [
        {{
            "area": "what they're confused about",
            "type": "misconception|gap|complexity",
            "severity": "low|medium|high"
        }}
    ],
    "readiness_signals": {{
        "canonical_ready": true/false,
        "deeper_exploration": true/false,
        "metaphor_switch_needed": true/false
    }}
}}
'''
        }
    
    async def extract_patterns(
        self, 
        user_input: str, 
        assistant_response: str,
        conversation_context: List[Dict[str, Any]] = None
    ) -> List[ConceptExtraction]:
        """Extract patterns from a conversation exchange."""
        
        try:
            # Extract concepts and relationships
            concept_data = await self._extract_concepts(user_input, assistant_response)
            
            # Analyze metaphors
            metaphor_data = await self._analyze_metaphors(user_input, assistant_response)
            
            # Analyze learning progression
            context_str = self._format_context(conversation_context or [])
            progression_data = await self._analyze_progression(
                user_input, assistant_response, context_str
            )
            
            # Combine into ConceptExtraction objects
            extractions = []
            
            if concept_data and "concepts" in concept_data:
                for concept_info in concept_data["concepts"]:
                    extraction = ConceptExtraction(
                        concept=concept_info.get("concept", ""),
                        domain=concept_info.get("domain", "unknown"),
                        metaphors=concept_info.get("metaphors", []),
                        confidence=concept_info.get("confidence", 0.5),
                        context=concept_info.get("context", ""),
                        relationships=concept_info.get("relationships", [])
                    )
                    extractions.append(extraction)
            
            # If no concepts extracted, create a minimal one
            if not extractions:
                extractions.append(ConceptExtraction(
                    concept=self._extract_main_topic(user_input, assistant_response),
                    domain="general",
                    confidence=0.3,
                    context=f"From user query: {user_input[:100]}..."
                ))
            
            return extractions
            
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            return []
    
    async def _extract_concepts(self, user_input: str, assistant_response: str) -> Optional[Dict[str, Any]]:
        """Extract concepts using LLM."""
        prompt = self.extraction_templates["concept_extraction"].format(
            user_input=user_input,
            assistant_response=assistant_response
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self._get_model_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse concept extraction JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Concept extraction failed: {e}")
            return None
    
    async def _analyze_metaphors(self, user_input: str, assistant_response: str) -> Optional[Dict[str, Any]]:
        """Analyze metaphors in the conversation."""
        prompt = self.extraction_templates["metaphor_analysis"].format(
            user_input=user_input,
            assistant_response=assistant_response
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self._get_model_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
                
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Metaphor analysis failed: {e}")
            return None
    
    async def _analyze_progression(self, user_input: str, assistant_response: str, context: str) -> Optional[Dict[str, Any]]:
        """Analyze learning progression."""
        prompt = self.extraction_templates["learning_progression"].format(
            context=context,
            user_input=user_input,
            assistant_response=assistant_response
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self._get_model_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
                
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Progression analysis failed: {e}")
            return None
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format conversation context for analysis."""
        if not context:
            return "No prior context"
        
        formatted = []
        for i, msg in enumerate(context[-5:]):  # Last 5 messages
            if isinstance(msg, dict):
                user_part = msg.get("user", "")
                assistant_part = msg.get("assistant", "")
                formatted.append(f"Exchange {i+1}:\nUser: {user_part}\nAssistant: {assistant_part}")
        
        return "\n\n".join(formatted)
    
    def _extract_main_topic(self, user_input: str, assistant_response: str) -> str:
        """Extract main topic as fallback when LLM extraction fails."""
        # Simple heuristic - look for capitalized words or quoted concepts
        text = f"{user_input} {assistant_response}"
        
        # Look for quoted concepts
        quoted = re.findall(r'"([^"]+)"', text)
        if quoted:
            return quoted[0]
        
        # Look for capitalized multi-word terms
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if capitalized:
            return capitalized[0]
        
        # Extract key words (very basic)
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        common_words = {'this', 'that', 'with', 'have', 'they', 'from', 'what', 'when', 'where', 'like'}
        key_words = [w for w in words if w not in common_words]
        
        if key_words:
            return key_words[0]
        
        return "general_concept"
    
    async def detect_metaphor_connections(
        self, 
        concepts: List[str], 
        conversation_text: str
    ) -> List[MetaphorsConnection]:
        """Detect connections between concepts via metaphors."""
        
        if len(concepts) < 2:
            return []
        
        prompt = f'''
Analyze these concepts from the conversation and find metaphorical connections:

CONCEPTS: {", ".join(concepts)}
CONVERSATION: {conversation_text}

Find where metaphors connect these concepts across domains. Return JSON:
{{
    "connections": [
        {{
            "source_concept": "concept A",
            "target_concept": "concept B",
            "metaphor": "the connecting metaphor",
            "domains": ["domain1", "domain2"],
            "strength": 0.8,
            "bidirectional": true
        }}
    ]
}}
'''
        
        try:
            response = await self.client.chat.completions.create(
                model=self._get_model_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            data = json.loads(content)
            
            connections = []
            for conn_data in data.get("connections", []):
                connection = MetaphorsConnection(
                    source_concept=conn_data.get("source_concept", ""),
                    target_concept=conn_data.get("target_concept", ""),
                    metaphor=conn_data.get("metaphor", ""),
                    domains=conn_data.get("domains", []),
                    strength=conn_data.get("strength", 0.5),
                    bidirectional=conn_data.get("bidirectional", True)
                )
                connections.append(connection)
            
            return connections
            
        except Exception as e:
            logger.error(f"Metaphor connection detection failed: {e}")
            return []
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about pattern extraction performance."""
        return {
            "templates_loaded": len(self.extraction_templates),
            "model": self.llm_config.model
        }