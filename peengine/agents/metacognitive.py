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

    async def generate_seed(self, session: Session) -> SeedDiscovery:
        """Generate exploration seed using LLM intelligence."""

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

        prompt = self._format_template(seed_template, context)

        try:
            response = await self.client.chat.completions.create(
                model=self._get_deployment_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,  # Creative but focused
                max_tokens=400,
            )

            # Parse JSON response
            seed_text = response.choices[0].message.content.strip()

            import re
            json_match = re.search(r"\{.*\}", seed_text, re.DOTALL)
            if json_match:
                seed_data = json.loads(json_match.group())
                seeds = seed_data.get("seeds", [])

                if seeds:
                    # Return the highest priority seed
                    best_seed = max(
                        seeds, key=lambda x: x.get("priority", 0.5))
                    return SeedDiscovery(
                        concept=best_seed.get("concept", "New exploration"),
                        rationale=best_seed.get(
                            "rationale", "Continue exploring"),
                        suggested_questions=best_seed.get(
                            "suggested_questions", []),
                    )

            # Fallback seed
            return SeedDiscovery(
                concept=f"Deeper exploration of {session.topic}",
                rationale="Continue building understanding",
                suggested_questions=[
                    f"What aspects of {session.topic} intrigue you most?"],
            )

        except Exception as e:
            logger.error(f"Seed generation failed: {e}")
            return SeedDiscovery(
                concept=f"Continue exploring {session.topic}",
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
