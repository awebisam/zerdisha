"""Metacognitive Agent - Monitors sessions and adjusts system behavior."""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

from openai import AsyncAzureOpenAI

from ..models.config import LLMConfig
from ..models.graph import Session, ConceptExtraction, SeedDiscovery

logger = logging.getLogger(__name__)


class MetacognitiveAgent:
    """Monitors learning sessions and provides meta-level adjustments."""
    
    def __init__(self, llm_config: LLMConfig):
        self.llm_config = llm_config
        
        # Initialize Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_key=llm_config.azure_openai_key,
            azure_endpoint=llm_config.azure_openai_endpoint,
            api_version=llm_config.azure_openai_api_version
        )
        
        # Analysis templates
        self.analysis_templates = self._load_analysis_templates()
        
        # Session monitoring state
        self.session_flags = {}
        self.adjustment_history = {}
    
    def _get_deployment_name(self) -> str:
        """Get the deployment name for metacognitive analysis."""
        return self.llm_config.azure_openai_deployment_name
    
    async def initialize(self) -> None:
        """Initialize the metacognitive agent."""
        logger.info("Metacognitive Agent initialized")
    
    def _load_analysis_templates(self) -> Dict[str, str]:
        """Load templates for metacognitive analysis."""
        return {
            "session_analysis": '''
Analyze this learning session for metacognitive patterns:

SESSION INFO:
- Topic: {topic}
- Duration: {duration_minutes} minutes
- Total exchanges: {total_exchanges}
- Concepts extracted: {concepts_count}

RECENT CONVERSATION:
{recent_messages}

CONCEPT EXTRACTIONS:
{extractions}

Analyze for:
1. Metaphor lock-in (same metaphor used too long)
2. Topic drift (wandering from main topic)
3. Stagnation (repeating same level without progress)
4. Readiness for canonical knowledge
5. Curiosity patterns and engagement level

Return JSON:
{{
    "flags": [
        {{
            "type": "metaphor_lock|topic_drift|stagnation|ready_for_canonical|low_engagement",
            "severity": "low|medium|high",
            "evidence": "what indicates this",
            "recommendation": "what to do about it"
        }}
    ],
    "insights": [
        {{
            "observation": "what you noticed",
            "impact": "how it affects learning",
            "suggestion": "how to optimize"
        }}
    ],
    "persona_adjustments": {{
        "metaphor_style": "adjust how metaphors are used",
        "question_depth": "adjust question complexity",
        "topic_focus": "guide topic boundaries",
        "pace": "adjust exploration pace"
    }},
    "suggested_commands": ["command1", "command2"]
}}
''',
            "seed_generation": '''
Based on this learning session, generate exploration seeds for future discovery:

SESSION SUMMARY:
- Topic: {topic} 
- Concepts explored: {concepts}
- Domains touched: {domains}
- Current understanding depth: {depth_level}

KNOWLEDGE GAPS IDENTIFIED:
{gaps}

Generate seeds for:
1. Unexplored connections between current concepts
2. Adjacent domains that could provide insights
3. Deeper layers of current topics
4. Cross-pollination opportunities

Return JSON:
{{
    "seeds": [
        {{
            "concept": "seed concept name",
            "discovery_type": "connection_gap|adjacent_domain|deeper_layer|cross_pollination",
            "rationale": "why this seed is valuable",
            "related_concepts": ["concept1", "concept2"],
            "suggested_questions": ["question1", "question2"],
            "priority": 0.8
        }}
    ]
}}
''',
            "final_session_analysis": '''
Provide final analysis for this completed learning session:

SESSION SUMMARY:
{session_summary}

FULL CONVERSATION TRAJECTORY:
{full_conversation}

CONCEPTS AND CONNECTIONS CREATED:
{graph_changes}

Assess:
1. Overall learning trajectory quality
2. Curiosity fulfillment vs. remaining open threads
3. Metaphor effectiveness across the session
4. Knowledge graph coherence and growth
5. Recommendations for next session

Return JSON:
{{
    "trajectory_quality": {{
        "score": 0.8,
        "strengths": ["strength1", "strength2"],
        "weaknesses": ["weakness1", "weakness2"]
    }},
    "curiosity_fulfillment": {{
        "fulfilled_aspects": ["aspect1", "aspect2"],
        "open_threads": ["thread1", "thread2"],
        "closure_quality": "high|medium|low"
    }},
    "metaphor_effectiveness": {{
        "successful_metaphors": [{"metaphor": "name", "effectiveness": 0.9}],
        "failed_metaphors": [{"metaphor": "name", "issue": "reason"}],
        "metaphor_diversity_score": 0.7
    }},
    "graph_coherence": {{
        "new_nodes_quality": "high|medium|low",
        "connection_strength": 0.8,
        "domain_integration": "good|fair|poor"
    }},
    "recommendations": {{
        "next_session_focus": "what to explore next",
        "learning_mode_adjustments": "how to adjust approach",
        "knowledge_gaps": ["gap1", "gap2"]
    }}
}}
'''
        }
    
    async def analyze_session(
        self, 
        session: Session, 
        recent_extractions: List[ConceptExtraction],
        ca_reasoning: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze current session state and provide adjustments."""
        
        try:
            # Format session data for analysis
            duration_minutes = 0
            if session.end_time:
                duration_minutes = (session.end_time - session.start_time).total_seconds() / 60
            elif session.start_time:
                duration_minutes = (datetime.utcnow() - session.start_time).total_seconds() / 60
            
            recent_messages = self._format_recent_messages(session.messages[-10:])
            extractions_text = self._format_extractions(recent_extractions)
            
            prompt = self.analysis_templates["session_analysis"].format(
                topic=session.topic,
                duration_minutes=round(duration_minutes, 1),
                total_exchanges=len(session.messages),
                concepts_count=len(session.nodes_created),
                recent_messages=recent_messages,
                extractions=extractions_text
            )
            
            response = await self.client.chat.completions.create(
                model=self._get_deployment_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1200
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            analysis = json.loads(content)
            
            # Store flags for session tracking
            self.session_flags[session.id] = analysis.get("flags", [])
            
            # Track adjustments
            if analysis.get("persona_adjustments"):
                self.adjustment_history[session.id] = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "adjustments": analysis["persona_adjustments"]
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Session analysis failed: {e}")
            return {
                "flags": [],
                "insights": [{"observation": "Analysis temporarily unavailable", "impact": "minimal", "suggestion": "continue session"}],
                "persona_adjustments": {},
                "suggested_commands": []
            }
    
    async def generate_seed(self, session: Session) -> SeedDiscovery:
        """Generate a new exploration seed based on session state."""
        
        try:
            # Gather session info
            concepts = self._extract_concepts_from_session(session)
            domains = self._extract_domains_from_session(session)
            depth_level = self._assess_depth_level(session)
            gaps = self._identify_knowledge_gaps(session)
            
            prompt = self.analysis_templates["seed_generation"].format(
                topic=session.topic,
                concepts=", ".join(concepts[:10]),  # Limit for prompt size
                domains=", ".join(domains),
                depth_level=depth_level,
                gaps=gaps
            )
            
            response = await self.client.chat.completions.create(
                model=self._get_deployment_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,  # Higher creativity for seed generation
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            data = json.loads(content)
            seeds = data.get("seeds", [])
            
            if seeds:
                # Return highest priority seed
                best_seed = max(seeds, key=lambda s: s.get("priority", 0))
                return SeedDiscovery(
                    concept=best_seed.get("concept", "unexplored_connection"),
                    discovery_type=best_seed.get("discovery_type", "connection_gap"),
                    rationale=best_seed.get("rationale", "Potential for new insights"),
                    related_concepts=best_seed.get("related_concepts", []),
                    suggested_questions=best_seed.get("suggested_questions", []),
                    priority=best_seed.get("priority", 0.5)
                )
            else:
                # Fallback seed
                return SeedDiscovery(
                    concept="deeper_exploration",
                    discovery_type="deeper_layer",
                    rationale="Continue exploring current topic in more depth",
                    related_concepts=concepts[:3],
                    suggested_questions=[f"What assumptions about {session.topic} haven't we questioned?"],
                    priority=0.5
                )
                
        except Exception as e:
            logger.error(f"Seed generation failed: {e}")
            return SeedDiscovery(
                concept="exploration_seed",
                discovery_type="general",
                rationale="Continue current exploration",
                suggested_questions=["What aspect of this topic intrigues you most?"],
                priority=0.3
            )
    
    async def finalize_session(self, session: Session) -> Dict[str, Any]:
        """Perform final analysis when session ends."""
        
        try:
            session_summary = {
                "topic": session.topic,
                "duration_minutes": (session.end_time - session.start_time).total_seconds() / 60 if session.end_time else 0,
                "exchanges": len(session.messages),
                "concepts_created": len(session.nodes_created),
                "connections_made": len(session.edges_created)
            }
            
            full_conversation = self._format_full_conversation(session.messages)
            graph_changes = self._format_graph_changes(session)
            
            prompt = self.analysis_templates["final_session_analysis"].format(
                session_summary=json.dumps(session_summary, indent=2),
                full_conversation=full_conversation[:3000],  # Truncate for token limits
                graph_changes=graph_changes
            )
            
            response = await self.client.chat.completions.create(
                model=self._get_deployment_name(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Final session analysis failed: {e}")
            return {
                "trajectory_quality": {"score": 0.5, "strengths": [], "weaknesses": []},
                "curiosity_fulfillment": {"fulfilled_aspects": [], "open_threads": [], "closure_quality": "medium"},
                "recommendations": {"next_session_focus": "Continue exploration"}
            }
    
    def _format_recent_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format recent messages for analysis."""
        formatted = []
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                user_part = msg.get("user", "")
                assistant_part = msg.get("assistant", "")
                formatted.append(f"Exchange {i+1}:\nUser: {user_part}\nAssistant: {assistant_part}")
        
        return "\n\n".join(formatted[-5:])  # Last 5 exchanges
    
    def _format_extractions(self, extractions: List[ConceptExtraction]) -> str:
        """Format concept extractions for analysis."""
        if not extractions:
            return "No recent extractions"
        
        formatted = []
        for extraction in extractions:
            formatted.append(f"- {extraction.concept} ({extraction.domain}): {extraction.context}")
        
        return "\n".join(formatted)
    
    def _format_full_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """Format full conversation for final analysis."""
        formatted = []
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                user_part = msg.get("user", "")[:200]  # Truncate long messages
                assistant_part = msg.get("assistant", "")[:200]
                formatted.append(f"{i+1}. U: {user_part}\n   A: {assistant_part}")
        
        return "\n".join(formatted)
    
    def _format_graph_changes(self, session: Session) -> str:
        """Format graph changes for analysis."""
        return f"Created {len(session.nodes_created)} nodes and {len(session.edges_created)} connections"
    
    def _extract_concepts_from_session(self, session: Session) -> List[str]:
        """Extract concept names from session messages."""
        # Simple extraction - in real implementation would use the graph
        concepts = []
        for msg in session.messages:
            if isinstance(msg, dict):
                text = f"{msg.get('user', '')} {msg.get('assistant', '')}"
                # Basic concept extraction - look for quoted terms
                import re
                quoted_terms = re.findall(r'"([^"]+)"', text)
                concepts.extend(quoted_terms)
        
        return list(set(concepts))[:10]  # Unique concepts, limited
    
    def _extract_domains_from_session(self, session: Session) -> List[str]:
        """Extract domains discussed in session."""
        # Heuristic based on topic and common domain keywords
        domains = [session.topic.split()[0].lower() if session.topic else "general"]
        
        # Look for domain indicators in messages
        domain_keywords = {
            "physics": ["force", "energy", "particle", "quantum", "relativity"],
            "biology": ["evolution", "cell", "organism", "genetics"],
            "chemistry": ["molecule", "reaction", "compound"],
            "mathematics": ["equation", "function", "theorem", "proof"],
            "computer_science": ["algorithm", "data", "programming", "code"]
        }
        
        text_content = " ".join([
            f"{msg.get('user', '')} {msg.get('assistant', '')}"
            for msg in session.messages
            if isinstance(msg, dict)
        ]).lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_content for keyword in keywords):
                domains.append(domain)
        
        return list(set(domains))
    
    def _assess_depth_level(self, session: Session) -> str:
        """Assess current depth level of understanding."""
        # Simple heuristic based on session length and complexity
        exchange_count = len(session.messages)
        
        if exchange_count < 3:
            return "surface"
        elif exchange_count < 10:
            return "developing"
        elif exchange_count < 20:
            return "intermediate"
        else:
            return "deep"
    
    def _identify_knowledge_gaps(self, session: Session) -> str:
        """Identify potential knowledge gaps from session."""
        # Placeholder - would analyze conversation for confusion indicators
        return "Analysis of conversation patterns suggests potential gaps in foundational concepts"
    
    def get_session_flags(self, session_id: str) -> List[Dict[str, Any]]:
        """Get current flags for a session."""
        return self.session_flags.get(session_id, [])
    
    def get_adjustment_history(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get adjustment history for a session."""
        return self.adjustment_history.get(session_id)