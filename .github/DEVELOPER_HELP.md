# PEEngine developer help (for AI assistants)

This is a quick, self-serve guide for working on this repo efficiently. It covers structure, commands, env, and common pitfalls so you can jump straight into useful changes and tests.

## TL;DR

- Entry point: `peengine/cli.py` (Typer app) exposes commands: `start`, `review`, `status`, `init`, `import-graphs`, `start-fresh`
- Orchestrator: `peengine/core/orchestrator.py` wires CA/PD/MA, DBs, embeddings, analytics
- Models: `peengine/models/graph.py` (Node/Edge/Session/Vector), `peengine/models/config.py` (Settings)
- Datastores: Neo4j (graph) + MongoDB (session & transcripts). Both must be running
- LLM: Azure OpenAI via Settings. Requires valid Azure env vars and a deployment name
- Tests live in `tests/`. Run with `pytest -q`

## Repo map (what to change where)

- peengine/cli.py
  - Typer CLI, interactive session loop, display helpers, adaptive error messaging
- peengine/core/orchestrator.py
  - Main engine: session lifecycle, CA/PD/MA pipeline, map/gap/seed commands, persistence
- peengine/core/embeddings.py, analytics.py
  - Vector creation and gap scoring; session reviews/analytics
- peengine/agents/
  - conversational.py, pattern_detector.py, metacognitive.py (async initialize + calls via orchestrator)
- peengine/database/
  - neo4j_client.py: graph CRUD, batch fetch, index creation, clear
  - mongodb_client.py: sessions/conversations storage, queries, clear
- peengine/tools/
  - import_knowledge_graphs.py: import domains and connections JSON
  - clear_databases.py: wipe Mongo + Neo4j (used by `start-fresh`)
- knowledge_graphs/
  - seed/domain and connection JSONs for import
- docs/
  - Testing, PRD, manual testing, persona

## CLI usage cheatsheet

- Initialize config and check basics
  - peengine init
- Start an exploration session
  - peengine start "Quantum Mechanics"
  - Interactive commands: /map, /gapcheck, /seed, /end, /help
- Import existing graphs (from `knowledge_graphs/`)
  - peengine import-graphs
- Status (connectivity and components)
  - peengine status
- Start from a clean slate (DESTRUCTIVE)
  - peengine start-fresh

Tip: pass `-v/--verbose` on commands that support it to enable richer logging.

## Environment variables (Settings)

Define these in `.env` (loaded via pydantic-settings):

- Azure OpenAI
  - AZURE_OPENAI_KEY
  - AZURE_OPENAI_ENDPOINT (like https://<resource>.openai.azure.com)
  - AZURE_OPENAI_DEPLOYMENT_NAME (deployment to use for chat)
  - AZURE_OPENAI_MODEL_NAME (model name associated with deployment)
  - AZURE_OPENAI_API_VERSION (default 2024-12-01-preview)
- Optional Azure AI Foundry fallback
  - AZURE_AI_FOUNDRY_API_KEY, AZURE_AI_FOUNDRY_ENDPOINT
- Model selection
  - PRIMARY_MODEL, FALLBACK_MODEL, PATTERN_MODEL
- Neo4j
  - NEO4J_URI (default bolt://localhost:7687)
  - NEO4J_USER (default neo4j)
  - NEO4J_PASSWORD
  - NEO4J_DATABASE (default neo4j)
- MongoDB
  - MONGODB_URI (default mongodb://localhost:27017)
  - MONGODB_DATABASE (default socratic_lab)
- App
  - LOG_LEVEL (default INFO)
  - PERSONA_PATH (default docs/external/persona.md)
  - KNOWLEDGE_GRAPHS_PATH (default knowledge_graphs/)

Note: README mentions GPT-5; config defaults target gpt-4.x. Ensure your Azure deployment name matches a real, accessible deployment.

## Local dev setup

- With Docker (recommended for DBs)
  - ./setup-docker.sh (or docker-compose.yml) to bring up Neo4j + MongoDB
  - Create `.env` with the vars above
  - pip install -e .
- Manual
  - Ensure Neo4j and MongoDB are running locally and accessible
  - Create `.env`
  - pip install -e .

## Tests

- Run all tests
  - pytest -q
- Useful targets
  - tests/test_cli.py, tests/test_orchestrator.py, tests/test_embeddings.py
- Style/type tools (pyproject.toml)
  - black, flake8, mypy (dev extras)

## Common pitfalls and fixes

- LLM/credentials
  - Symptom: initialization or chat calls fail
  - Check: AZURE_OPENAI_* envs set; deployment exists; endpoint correct
- Neo4j connection/index errors
  - Ensure Bolt URL and credentials; database name exists
  - Index creation warnings are OK (already exists)
- MongoDB connection/index errors
  - Ensure server running and MONGODB_URI correct
- CLI hangs or exits early
  - Ctrl-C shows advice to use /end; otherwise restart and use `/end` to persist summary
- Gap check surprises
  - Requires some conversational context and concept detection; talk about a specific concept first

## Making changes safely

- Keep public CLI behavior and Typer signatures stable unless tests are updated
- Maintain async patterns and timeouts around LLM calls to avoid stalls
- Prefer idempotent DB ops (MERGE on Neo4j) and batch fetch where available
- Validate incoming data and degrade gracefully; tests expect friendly fallbacks

## Quick contract reminders (orchestrator APIs)

- ExplorationEngine.initialize(): connects DBs, initializes agents (async)
- start_session(topic, title?): creates Mongo session + Neo4j summary
- process_user_input(text): returns { message, session_id, new_concepts, ma_insights, suggested_commands }
- execute_command("map"|"gapcheck"|"seed"|"end"): returns dict payload for display helpers
- end_session(): returns session summary and clears current_session

Edge cases to consider
- No active session; empty conversation; DB intermittent failures; LLM timeouts; missing vectors; large messages (truncate)

## Where to put new things

- New CLI commands: add in `peengine/cli.py`, implement engine method in orchestrator or services
- New analytics: `peengine/core/analytics.py` and wire via orchestrator
- New imports/tools: `peengine/tools/`
- New graph ops: extend `peengine/database/neo4j_client.py`

## Troubleshooting quick checks

- peengine init
- peengine status
- If still failing, inspect `peengine.log` (when verbose logging is enabled)

---

This doc is for fast navigation and productive edits. If something here goes stale, update it alongside the change.
