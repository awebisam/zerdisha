"""Metacognitive Agent - Monitors sessions and adjusts system behavior."""

import logging
import json
from typing import Dict, List, Any
from datetime import datetime

from openai import AsyncAzureOpenAI

from ..models.config import LLMConfig
from ..core.prompts import (
    MA_SESSION_ANALYSIS,
    MA_SEED_GENERATION,
    MA_FINAL_SESSION_ANALYSIS,
)
from ..models.graph import Session, ConceptExtraction, SeedDiscovery
from ..models.responses import SeedResponse, SeedItem
from ..core.json_utils import safe_load_json

logger = logging.getLogger(__name__)


class MetacognitiveAgent:
    """Monitors learning sessions and provides meta-level adjustments using an LLM-first approach."""

    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        self.client = AsyncAzureOpenAI(
            api_key=llm_config.azure_openai_key,
            azure_endpoint=llm_config.azure_openai_endpoint,
            api_version=llm_config.azure_openai_api_version
        )
        self.session_flags = {}

    def _get_deployment_name(self) -> str:
        """Get the deployment name for metacognitive analysis."""
        return self.llm_config.azure_openai_deployment_name

    async def initialize(self) -> None:
        """Initialize the metacognitive agent."""
        logger.info("Metacognitive Agent initialized with LLM-first approach")

    def _load_analysis_templates(self) -> Dict[str, str]:
        """Load templates for metacognitive analysis from centralized prompt library."""
        return {
            "session_analysis": MA_SESSION_ANALYSIS,
            "seed_generation": MA_SEED_GENERATION,
            "final_session_analysis": MA_FINAL_SESSION_ANALYSIS,
        }

    async def analyze_session(
        self,
        session: Session,
        extractions: List[ConceptExtraction],
        ca_reasoning: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze session for metacognitive patterns with enhanced LLM intelligence."""

        # Enhanced metaphor tracking with pattern analysis
        metaphor_analysis = await self._analyze_metaphor_patterns(session.messages, extractions)

        # Build comprehensive analysis context
        context = {
            "topic": session.topic,
            "duration_minutes": (datetime.utcnow() - session.start_time).total_seconds() / 60,
            "total_exchanges": len(session.messages),
            "concepts_count": len(extractions),
            # Last 3 exchanges (max 6 items: 3 user/assistant pairs)
            "recent_messages": self._format_recent_messages(session.messages[-6:]),
            "extractions": self._format_extractions(extractions),
            "metaphor_usage": self._track_metaphor_usage(session.messages, extractions),
            "metaphor_analysis": metaphor_analysis,
        }

        # Use enhanced analysis template
        templates = self._load_analysis_templates()
        analysis_template = templates["session_analysis"]
        prompt = self._format_template(analysis_template, context)

        try:
            response = await self.client.chat.completions.create(
                model=self._get_deployment_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,  # Balanced analysis
                max_tokens=700,  # Increased for richer analysis
            )

            analysis_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            import re
            json_match = re.search(r"\{.*\}", analysis_text, re.DOTALL)
            if not json_match:
                logger.warning("Could not parse JSON from MA analysis")
                return {"error": "Could not parse JSON from MA analysis", "flags": [], "insights": [], "persona_adjustments": {}}

            analysis = json.loads(json_match.group())

            # Validate expected top-level keys exist
            expected_keys = {"flags", "insights", "persona_adjustments"}
            if not any(k in analysis for k in expected_keys):
                return {"error": "Invalid analysis structure", **analysis}

            # Enhance analysis with additional intelligent insights
            analysis = await self._enhance_analysis_insights(analysis, context)

            # Store flags for session tracking
            self.session_flags[session.id] = analysis.get("flags", [])

            logger.info(
                f"Enhanced MA analysis complete: {len(analysis.get('flags', []))} flags, {len(analysis.get('insights', []))} insights"
            )
            return analysis

        except Exception as e:
            logger.error(f"MA analysis failed: {e}")
            return {"error": f"MA analysis failed: {e}", "flags": [], "insights": [], "persona_adjustments": {}}

    async def _analyze_metaphor_patterns(self, messages: List[Dict], extractions: List[ConceptExtraction]) -> Dict[str, Any]:
        """Analyze metaphor patterns with LLM intelligence."""

        # Extract recent conversation text
        conversation_text = "\n".join([
            f"User: {msg.get('user', '')}\nAssistant: {msg.get('assistant', '')}"
            for msg in messages[-8:]  # Last 4 exchanges
        ])

        # Extract metaphors from extractions
        all_metaphors = []
        for extraction in extractions:
            all_metaphors.extend(extraction.metaphors)

        prompt = f"""
Analyze the metaphor usage patterns in this learning conversation:

CONVERSATION:
{conversation_text}

EXTRACTED METAPHORS:
{', '.join(all_metaphors)}

Analyze for:
1. Metaphor diversity - are they using varied metaphorical domains?
2. Metaphor lock-in - repetitive use of same metaphorical framework
3. Metaphor effectiveness - which metaphors seem to help vs. hinder understanding
4. Metaphor evolution - how metaphors change over the conversation
5. Missing metaphorical opportunities - concepts that could benefit from new metaphors

Return JSON:
{{
    "diversity_score": 0.7,
    "lock_in_detected": true,
    "dominant_metaphor": "water flow",
    "metaphor_domains": ["nature", "mechanics", "building"],
    "effectiveness_assessment": "High effectiveness for concrete concepts, struggling with abstract ones",
    "evolution_pattern": "Started with mechanical metaphors, moving toward organic ones",
    "suggested_new_domains": ["music", "dance", "cooking"],
    "lock_in_evidence": "Used 'flow' metaphor 5 times in last 3 exchanges"
}}
"""

        try:
            response = await self.client.chat.completions.create(
                model=self._get_deployment_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )

            analysis_text = response.choices[0].message.content.strip()
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)

            if json_match:
                return json.loads(json_match.group())
            else:
                return {"diversity_score": 0.5, "lock_in_detected": False}

        except Exception as e:
            logger.error(f"Metaphor pattern analysis failed: {e}")
            return {"diversity_score": 0.5, "lock_in_detected": False}

    async def _enhance_analysis_insights(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance analysis with additional intelligent insights."""

        # Add learning trajectory assessment
        if "learning_trajectory" not in analysis:
            trajectory_score = self._assess_learning_trajectory(context)
            analysis["learning_trajectory"] = {
                "progression_score": trajectory_score,
                "depth_level": "surface" if trajectory_score < 0.3 else "developing" if trajectory_score < 0.7 else "deep",
                "momentum": "building" if trajectory_score > 0.6 else "steady" if trajectory_score > 0.3 else "declining"
            }

        # Add curiosity health metrics
        if "curiosity_health" not in analysis:
            engagement_level = "high" if context.get(
                "total_exchanges", 0) > 5 else "medium" if context.get("total_exchanges", 0) > 2 else "low"
            analysis["curiosity_health"] = {
                "engagement_level": engagement_level,
                "question_quality": "stable",  # Could be enhanced with more analysis
                "exploration_breadth": "focused" if context.get("concepts_count", 0) < 3 else "expanding"
            }

        return analysis

    def _assess_learning_trajectory(self, context: Dict[str, Any]) -> float:
        """Assess learning trajectory quality."""
        # Simple heuristic based on conversation length and concept extraction
        exchanges = context.get("total_exchanges", 0)
        concepts = context.get("concepts_count", 0)

        if exchanges == 0:
            return 0.0

        # Basic trajectory score: concepts per exchange with diminishing returns
        base_score = min(concepts / max(exchanges, 1), 1.0)

        # Adjust for session duration (longer sessions with sustained engagement are better)
        duration = context.get("duration_minutes", 0)
        # Optimal around 30 minutes
        duration_factor = min(duration / 30.0, 1.2)

        return min(base_score * duration_factor, 1.0)

    def _format_recent_messages(self, messages: List[Dict]) -> str:
        """Format recent messages for analysis."""
        if not messages:
            return "No recent messages"

        formatted = []
        for msg in messages:
            user_msg = msg.get('user', '')
            assistant_msg = msg.get('assistant', '')
            formatted.append(f"User: {user_msg}\nAssistant: {assistant_msg}")

        return "\n---\n".join(formatted)

    def _format_extractions(self, extractions: List[ConceptExtraction]) -> str:
        """Format concept extractions for analysis."""
        if not extractions:
            return "No concept extractions"

        formatted = []
        for extraction in extractions:
            metaphors_str = ", ".join(
                extraction.metaphors) if extraction.metaphors else "None"
            formatted.append(
                f"Concept: {extraction.concept} | Domain: {extraction.domain} | Metaphors: {metaphors_str}")

        return "\n".join(formatted)

    def _track_metaphor_usage(self, messages: List[Dict], extractions: List[ConceptExtraction]) -> str:
        """Track metaphor usage patterns."""
        # Extract all metaphors from extractions
        all_metaphors = []
        for extraction in extractions:
            all_metaphors.extend(extraction.metaphors)

        if not all_metaphors:
            return "No metaphors detected"

        # Count metaphor frequency
        metaphor_counts = {}
        for metaphor in all_metaphors:
            metaphor_counts[metaphor] = metaphor_counts.get(metaphor, 0) + 1

        # Format for analysis
        usage_summary = []
        for metaphor, count in sorted(metaphor_counts.items(), key=lambda x: x[1], reverse=True):
            usage_summary.append(f"{metaphor}: {count} times")

        return "\n".join(usage_summary[:10])  # Top 10 most used

    async def generate_seed(self, session: Session, preferences: Dict[str, Any] | None = None) -> SeedDiscovery:
        """Generate exploration seed using LLM intelligence.

        preferences options (all optional):
        - discovery_type: one of connection_gap|adjacent_domain|deeper_layer|cross_pollination
        - concept: preferred concept focus to bias selection
        """

        # Build context for seed generation
        concepts = [msg.get("user", "") + " " + msg.get("assistant", "")
                    for msg in session.messages[-4:]]
        concepts_text = " ".join(concepts)

        # Identify domains from session
        domains = ["general"]  # Default domain

        templates = self._load_analysis_templates()
        seed_template = templates["seed_generation"]

        context = {
            "topic": session.topic,
            "concepts": concepts_text[:500],  # Limit length
            "domains": ", ".join(domains),
            "depth_level": "developing",  # Could be enhanced with actual assessment
            "gaps": "To be identified through analysis",
        }

        # If user prefers a discovery type or concept, lightly hint in prompt
        pref = preferences or {}
        hint_lines = []
        if isinstance(pref.get("discovery_type"), str):
            hint_lines.append(
                f"Preferred discovery_type: {pref['discovery_type']}")
        if isinstance(pref.get("concept"), str):
            hint_lines.append(f"Preferred concept focus: {pref['concept']}")

        hint_block = ("\nUser preferences:\n" +
                      "\n".join(hint_lines) + "\n") if hint_lines else "\n"
        prompt = self._format_template(seed_template, context) + hint_block

        try:
            # Prefer Responses API with explicit JSON schema for structure
            seed_schema = {
                "type": "object",
                "properties": {
                    "seeds": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "concept": {"type": "string"},
                                "discovery_type": {
                                    "type": "string",
                                    "enum": [
                                        "connection_gap",
                                        "adjacent_domain",
                                        "deeper_layer",
                                        "cross_pollination"
                                    ]
                                },
                                "rationale": {"type": "string"},
                                "related_concepts": {"type": "array", "items": {"type": "string"}},
                                "suggested_questions": {"type": "array", "items": {"type": "string"}},
                                "priority": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            },
                            "required": ["concept", "discovery_type", "rationale"]
                        }
                    }
                },
                "required": ["seeds"]
            }

            # Try Responses API first for clean JSON
            seeds: List[Dict[str, Any]] = []
            try:
                resp = await self.client.responses.create(
                    model=self._get_deployment_name(),
                    input=[{"role": "user", "content": prompt}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "seed_response", "schema": seed_schema}
                    },
                    temperature=0.6,
                    max_output_tokens=400,
                )
                if resp and getattr(resp, "output", None):
                    parts = getattr(resp.output[0], "content", None) or []
                    for part in parts:
                        if getattr(part, "type", "") == "output_text":
                            raw = getattr(part, "text", "")
                            data, err = safe_load_json(raw)
                            if err:
                                logger.warning(
                                    f"Seed JSON parse issue (responses): {err}")
                            if isinstance(data, dict):
                                seeds = data.get("seeds", []) or []
                            break
            except Exception as responses_err:
                logger.info(
                    f"Responses API failed for seeds, falling back to chat JSON mode: {responses_err}")

            # Fallback: Chat JSON mode if Responses didn't yield seeds
            if not seeds:
                chat = await self.client.chat.completions.create(
                    model=self._get_deployment_name(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=400,
                    response_format={"type": "json_object"},
                )
                content = (chat.choices[0].message.content or "").strip()
                data, err = safe_load_json(content)
                if err:
                    logger.warning(f"Seed JSON parse issue (chat): {err}")
                if isinstance(data, dict):
                    seeds = data.get("seeds", []) or []

            # Final fallback: plain chat, attempt substring JSON parse
            if not seeds:
                chat = await self.client.chat.completions.create(
                    model=self._get_deployment_name(),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=400,
                )
                text = (chat.choices[0].message.content or "").strip()
                data, err = safe_load_json(text)
                if err:
                    logger.warning(f"Seed JSON parse issue (plain): {err}")
                if isinstance(data, dict):
                    seeds = data.get("seeds", []) or []

            # Optional filtering by preferences
            if seeds and pref.get("discovery_type"):
                seeds = [s for s in seeds if s.get(
                    "discovery_type") == pref.get("discovery_type")] or seeds
            if seeds and pref.get("concept"):
                target = str(pref.get("concept")).lower()
                # Rank by whether concept substring appears, then priority

                def rank(s: Dict[str, Any]) -> tuple:
                    c = str(s.get("concept", "")).lower()
                    contains = 1 if target and target in c else 0
                    return (contains, float(s.get("priority", 0.5)))
                seeds = sorted(seeds, key=rank, reverse=True)

            # Validate with Pydantic for safety
            try:
                model = SeedResponse.model_validate({"seeds": seeds})
            except Exception as val_err:
                logger.warning(
                    f"SeedResponse validation failed, attempting to coerce: {val_err}")
                # Try to coerce discovery_type to allowed values when possible
                coerced: List[Dict[str, Any]] = []
                allowed = {"connection_gap", "adjacent_domain",
                           "deeper_layer", "cross_pollination"}
                for s in seeds or []:
                    dt = s.get("discovery_type")
                    if dt not in allowed:
                        # default to deeper_layer when unknown
                        s = {**s, "discovery_type": "deeper_layer"}
                    coerced.append(s)
                try:
                    model = SeedResponse.model_validate({"seeds": coerced})
                except Exception as val_err2:
                    logger.error(
                        f"SeedResponse validation failed after coercion: {val_err2}")
                    model = SeedResponse(seeds=[])

            if model.seeds:
                # Prefer highest priority after any user-target ranking has been applied
                def best_key(item: SeedItem):
                    return item.priority
                best_item = max(model.seeds, key=best_key)
                return SeedDiscovery(
                    concept=best_item.concept or f"Deeper exploration of {session.topic}",
                    discovery_type=best_item.discovery_type,
                    rationale=best_item.rationale or "Continue exploring",
                    related_concepts=list(best_item.related_concepts or []),
                    suggested_questions=list(
                        best_item.suggested_questions or []),
                    priority=float(best_item.priority),
                )

            # Fallback seed
            return SeedDiscovery(
                concept=f"Deeper exploration of {session.topic}",
                discovery_type=str(
                    pref.get("discovery_type") or "deeper_layer"),
                rationale="Continue building understanding",
                suggested_questions=[
                    f"What aspects of {session.topic} intrigue you most?"],
            )

        except Exception as e:
            logger.error(f"Seed generation failed: {e}")
            return SeedDiscovery(
                concept=f"Continue exploring {session.topic}",
                discovery_type=str((preferences or {}).get(
                    "discovery_type") or "deeper_layer"),
                rationale="Keep the exploration momentum going",
                suggested_questions=[
                    f"What new angle on {session.topic} would you like to explore?"],
            )

    async def finalize_session(self, session: Session) -> Dict[str, Any]:
        """Provide final session analysis using LLM intelligence."""

        # Build comprehensive session summary
        session_summary = {
            "topic": session.topic,
            "duration": (session.end_time - session.start_time).total_seconds() / 60 if session.end_time else 0,
            "total_exchanges": len(session.messages),
            "concepts_created": len(session.nodes_created),
            "connections_made": len(session.edges_created),
        }

        # Format conversation for analysis
        full_conversation = self._format_recent_messages(session.messages)

        # Graph changes summary
        graph_changes = f"Created {len(session.nodes_created)} concept nodes and {len(session.edges_created)} connections"

        templates = self._load_analysis_templates()
        final_template = templates["final_session_analysis"]

        context = {
            "session_summary": json.dumps(session_summary, indent=2),
            # Limit for token management
            "full_conversation": full_conversation[:1000],
            "graph_changes": graph_changes,
        }

        prompt = self._format_template(final_template, context)

        try:
            response = await self.client.chat.completions.create(
                model=self._get_deployment_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=500,
            )

            analysis_text = response.choices[0].message.content.strip()

            import re
            json_match = re.search(r"\{.*\}", analysis_text, re.DOTALL)
            if json_match:
                final_analysis = json.loads(json_match.group())
                logger.info("Generated comprehensive final session analysis")
                return final_analysis
            else:
                logger.warning("Could not parse final analysis JSON")
                return self._fallback_final_analysis(session_summary)

        except Exception as e:
            logger.error(f"Final session analysis failed: {e}")
            return self._fallback_final_analysis(session_summary)

    def _fallback_final_analysis(self, session_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback final analysis if LLM fails."""
        return {
            "trajectory_quality": {
                "score": 0.7,
                "strengths": ["Active engagement", "Concept exploration"],
                "weaknesses": ["Could explore deeper connections"]
            },
            "curiosity_fulfillment": {
                "fulfilled_aspects": ["Basic understanding", "Initial exploration"],
                "open_threads": ["Deeper connections", "Practical applications"],
                "closure_quality": "medium"
            },
            "recommendations": {
                "next_session_focus": f"Continue exploring {session_summary.get('topic', 'the topic')}",
                "learning_mode_adjustments": "Consider exploring connections to other domains",
                "knowledge_gaps": ["Deeper theoretical understanding", "Practical applications"]
            }
        }

    def _format_template(self, template: str, mapping: Dict[str, Any]) -> str:
        """Safely format a template that contains JSON braces by escaping all braces
        and then restoring placeholders for provided mapping keys.

        Example: turns all '{' -> '{{' and '}' -> '}}', then for each key 'topic'
        restores '{{topic}}' -> '{topic}' so .format works only on placeholders.
        """
        # Escape all braces first
        escaped = template.replace('{', '{{').replace('}', '}}')
        # Restore placeholders for keys in mapping
        for key in mapping.keys():
            escaped = escaped.replace('{{' + key + '}}', '{' + key + '}')
        try:
            return escaped.format(**mapping)
        except Exception:
            # As a last resort, return the unformatted template to avoid crashes
            return template
