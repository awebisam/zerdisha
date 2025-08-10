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
