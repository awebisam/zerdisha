# Node: Features

**ID:** `features`
**Type:** `Implementation`

This document lists the currently implemented features of the PEEngine.

## Implemented Features:

*   **Three-Agent System:** The CA, PD, and MA are implemented as separate classes.
*   **Orchestrator:** A central `ExplorationEngine` class manages the agents and databases.
*   **Database Integration:** Neo4j and MongoDB clients are fully integrated.
*   **CLI:** A functional Typer-based CLI with commands for managing sessions.
*   **Vector Embeddings:** The `EmbeddingService` can create and compare u-vectors and c-vectors.
*   **Knowledge Graph Import:** A tool for importing existing knowledge graphs is included.
*   **`/gapcheck` Command:** Fully functional gap analysis between user understanding (u-vectors) and canonical knowledge (c-vectors).
*   **Enhanced `/map` Command:** The map now displays both concepts and natural language descriptions of the relationships between them.
*   **Metacognitive (MA) Influence:** The MA can analyze session patterns (like metaphor lock-in) and provide persona adjustments that are applied to the Conversational Agent (CA) via the `/apply_ma` command.
*   **`/seed` Command:** The MA can generate new exploration seeds to help guide the user.
*   **Robust Testing Suite:** The project includes a comprehensive suite of manual and automated tests.