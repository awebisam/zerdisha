# Node: Architecture

**ID:** `architecture`
**Type:** `SystemDesign`

This document describes the technical architecture of the PEEngine, which is a direct manifestation of its core philosophy.

## Core Components:

*   **Conversational Agent (CA):** The user's Socratic guide.
*   **Pattern Detector (PD):** The agent that maps the conversation to the knowledge graph.
*   **Metacognitive Agent (MA):** The agent that monitors the learning process.

## Technical Stack:

*   **CLI:** Python with Typer and Rich.
*   **Databases:**
    *   **Neo4j:** For the knowledge graph.
    *   **MongoDB:** For session data.
*   **AI:** Azure OpenAI (GPT-5).
*   **Data Modeling:** Pydantic.
