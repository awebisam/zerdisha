"""Centralized prompt templates for agents (baseline v1)."""

# Metacognitive Agent templates
MA_SESSION_ANALYSIS = '''
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

METAPHOR USAGE TRACKING:
{metaphor_usage}

Analyze for:
1. Metaphor lock-in (same metaphor used repeatedly - look for patterns like "like a", "similar to", "imagine", "think of X as Y")
2. Topic drift (wandering from main topic)
3. Stagnation (repeating same level without progress)
4. Readiness for canonical knowledge
5. Curiosity patterns and engagement level

Pay special attention to metaphor patterns:
- Identify specific metaphors being used (e.g., "water flow", "building blocks", "journey", "dance")
- Count repetitions of the same metaphorical framework
- Assess if metaphor diversity is decreasing over time
- Look for signs that current metaphors are limiting rather than expanding understanding

Return JSON:
{
    "flags": [
        {
            "type": "metaphor_lock|topic_drift|stagnation|ready_for_canonical|low_engagement",
            "severity": "low|medium|high",
            "evidence": "what indicates this",
            "recommendation": "what to do about it"
        }
    ],
    "insights": [
        {
            "observation": "what you noticed",
            "impact": "how it affects learning",
            "suggestion": "how to optimize"
        }
    ],
    "metaphors_detected": [
        {
            "metaphor": "specific metaphor name or description",
            "frequency": "how often used",
            "effectiveness": "high|medium|low",
            "limiting_factor": "how it might be constraining thinking"
        }
    ],
    "persona_adjustments": {
        "metaphor_style": "adjust how metaphors are used - encourage diversity, introduce new domains",
        "question_depth": "adjust question complexity",
        "topic_focus": "guide topic boundaries",
        "pace": "adjust exploration pace",
        "metaphor_prompting": "specific instructions to encourage new metaphorical thinking"
    },
    "suggested_commands": ["command1", "command2"]
}
'''

MA_SEED_GENERATION = '''
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
{
    "seeds": [
        {
            "concept": "seed concept name",
            "discovery_type": "connection_gap|adjacent_domain|deeper_layer|cross_pollination",
            "rationale": "why this seed is valuable",
            "related_concepts": ["concept1", "concept2"],
            "suggested_questions": ["question1", "question2"],
            "priority": 0.8
        }
    ]
}
'''

MA_FINAL_SESSION_ANALYSIS = '''
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
{
    "trajectory_quality": {
        "score": 0.8,
        "strengths": ["strength1", "strength2"],
        "weaknesses": ["weakness1", "weakness2"]
    },
    "curiosity_fulfillment": {
        "fulfilled_aspects": ["aspect1", "aspect2"],
        "open_threads": ["thread1", "thread2"],
        "closure_quality": "high|medium|low"
    },
    "metaphor_effectiveness": {
        "successful_metaphors": [{"metaphor": "name", "effectiveness": 0.9}],
        "failed_metaphors": [{"metaphor": "name", "issue": "reason"}],
        "metaphor_diversity_score": 0.7
    },
    "graph_coherence": {
        "new_nodes_quality": "high|medium|low",
        "connection_strength": 0.8,
        "domain_integration": "good|fair|poor"
    },
    "recommendations": {
        "next_session_focus": "what to explore next",
        "learning_mode_adjustments": "how to adjust approach",
        "knowledge_gaps": ["gap1", "gap2"]
    }
}
'''

# Conversational Agent persona synthesis instruction
CA_PERSONA_SYNTHESIS = '''
You are helping synthesize persona adjustments for a Socratic learning guide. The goal is to intelligently integrate new behavioral adjustments with existing ones, resolving conflicts and creating coherent guidance.

CURRENT PERSONA ADJUSTMENTS:
{current_adjustments}

NEW ADJUSTMENTS TO INTEGRATE:
{new_adjustments}

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

Return only valid JSON with synthesized adjustments.
'''
