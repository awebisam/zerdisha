"""Pattern Detector - Extracts concepts and relationships from conversations.

Hardened JSON parsing:
- Strips code fences
- Extracts largest JSON object from mixed text
- Cleans common issues (smart quotes, trailing commas)
- Tries JSON mode response when supported, with fallback
"""

import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple

# Removed direct openai import; we use the shared AzureLLMClient wrapper now

from ..models.config import LLMConfig
from ..models.graph import ConceptExtraction, MetaphorsConnection
from ..core.llm_client import AzureLLMClient
from ..core.json_utils import safe_load_json

logger = logging.getLogger(__name__)


class PatternDetector:
    """Extracts conceptual patterns and relationships from conversations."""

    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config

        # Initialize shared Responses-first client
        self.llm = AzureLLMClient(
            api_key=llm_config.azure_openai_key,
            endpoint=llm_config.azure_openai_endpoint,
            api_version=llm_config.azure_openai_api_version,
            deployment=llm_config.azure_openai_deployment_name,
        )

        # Pattern extraction templates
        self.extraction_templates = self._load_extraction_templates()

    def _get_deployment_name(self) -> str:
        """Get the deployment name for pattern extraction (using efficient model)."""
        return self.llm_config.azure_openai_deployment_name  # Could use pattern_model deployment

    async def initialize(self) -> None:
        """Initialize the pattern detector."""
        logger.info("Pattern Detector initialized")

    # --------------------
    # JSON parsing helpers
    # --------------------

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        if not text:
            return text
        t = text.strip()
        if t.startswith("```json"):
            t = t[7:]
        if t.startswith("```"):
            t = t[3:]
        if t.endswith("```"):
            t = t[:-3]
        return t.strip()

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Apply lightweight fixes for common LLM JSON issues without overfitting.
        - Replace smart quotes with standard quotes
        - Remove trailing commas before closing } or ]
        - Normalize line endings
        """
        if not text:
            return text
        # Replace smart quotes
        text = (
            text.replace("\u201c", '"')
                .replace("\u201d", '"')
                .replace("\u2018", "'")
                .replace("\u2019", "'")
                .replace("“", '"')
                .replace("”", '"')
                .replace("‘", "'")
                .replace("’", "'")
        )
        # Remove trailing commas before closing braces/brackets
        text = re.sub(r",\s*(}\s*)", r"\1", text)
        text = re.sub(r",\s*(]\s*)", r"\1", text)
        # Normalize newlines
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()

    @staticmethod
    def _extract_json_substring(text: str) -> Optional[str]:
        """Attempt to extract the largest valid-looking JSON object or array substring.
        Uses a brace/bracket stack to find a balanced region. Returns substring or None.
        """
        if not text:
            return None
        # Find first '{' or '[' as start
        start_candidates = [i for i, ch in enumerate(text) if ch in '{[']
        for start in start_candidates:
            stack = []
            in_string = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == '\\':
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                else:
                    if ch == '"':
                        in_string = True
                        continue
                    if ch in '{[':
                        stack.append(ch)
                    elif ch in '}]':
                        if not stack:
                            break
                        open_ch = stack.pop()
                        if (open_ch == '{' and ch != '}') or (open_ch == '[' and ch != ']'):
                            break
                        if not stack:
                            # Balanced region found
                            return text[start:i + 1]
            # try next candidate
        return None

    @classmethod
    def _safe_load_json(cls, raw_content: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Try to load JSON from raw LLM content with multiple strategies.
        Returns (data, error_message).
        """
        if raw_content is None:
            return None, "No content"
        content = cls._strip_code_fences(raw_content)
        content = cls._clean_json_text(content)
        # First attempt
        try:
            return json.loads(content), None
        except Exception as e1:
            # Try to extract a JSON substring
            substring = cls._extract_json_substring(content)
            if substring:
                try:
                    return json.loads(cls._clean_json_text(substring)), None
                except Exception as e2:
                    # As a last resort, try a naive single->double quote conversion on substring
                    naive = substring
                    # Replace single quotes only when they appear as string delimiters: {'key': 'val'}
                    naive = re.sub(r"'([^'\\]*)'", r'"\1"', naive)
                    try:
                        return json.loads(cls._clean_json_text(naive)), None
                    except Exception as e3:
                        snippet = (
                            raw_content[:200] + '...') if len(raw_content) > 200 else raw_content
                        return None, f"JSON parse failed: {e1} | substring failed: {e2} | naive failed: {e3} | content snippet: {snippet}"
            else:
                snippet = (
                    raw_content[:200] + '...') if len(raw_content) > 200 else raw_content
                return None, f"JSON parse failed: {e1} | no balanced substring | content snippet: {snippet}"

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
                    concept=self._extract_main_topic(
                        user_input, assistant_response),
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
            # Try JSON mode first (if supported by deployment), then fallback
            concept_schema = {
                "type": "object",
                "properties": {
                    "concepts": {"type": "array"},
                    "metaphor_connections": {"type": "array"},
                    "domain_info": {"type": "object"},
                },
            }
            content = await self.llm.json_response(prompt, schema=concept_schema, temperature=0.3, max_tokens=1000)
            data, err = safe_load_json(content)
            if err:
                logger.error(f"Failed to parse concept extraction JSON: {err}")
            return data
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
            metaphor_schema = {
                "type": "object",
                "properties": {
                    "metaphors": {"type": "array"},
                    "cross_domain_links": {"type": "array"},
                },
            }
            content = await self.llm.json_response(prompt, schema=metaphor_schema, temperature=0.3, max_tokens=800)
            data, err = safe_load_json(content)
            if err:
                logger.error(f"Metaphor analysis failed: {err}")
            return data
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
            progression_schema = {
                "type": "object",
                "properties": {
                    "understanding_depth": {"type": "object"},
                    "breakthroughs": {"type": "array"},
                    "confusions": {"type": "array"},
                    "readiness_signals": {"type": "object"},
                },
            }
            content = await self.llm.json_response(prompt, schema=progression_schema, temperature=0.3, max_tokens=600)
            data, err = safe_load_json(content)
            if err:
                logger.error(f"Progression analysis failed: {err}")
            return data
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
                formatted.append(
                    f"Exchange {i+1}:\nUser: {user_part}\nAssistant: {assistant_part}")

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
        common_words = {'this', 'that', 'with', 'have',
                        'they', 'from', 'what', 'when', 'where', 'like'}
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
            connections_schema = {
                "type": "object",
                "properties": {
                    "connections": {"type": "array"},
                },
            }
            content = await self.llm.json_response(prompt, schema=connections_schema, temperature=0.3, max_tokens=600)
            data, err = safe_load_json(content)
            if err:
                logger.error(
                    f"Metaphor connection detection failed to parse: {err}")
                return []

            connections = []
            for conn_data in (data or {}).get("connections", []):
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
            "model": self.llm_config.primary_model
        }
