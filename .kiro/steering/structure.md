# Project Structure & Organization

## Root Directory Layout

```
peengine/                    # Main package directory
├── __init__.py
├── cli.py                   # Main CLI entry point with Typer commands
├── agents/                  # Three-agent architecture
│   ├── conversational.py   # Socratic dialogue agent (CA)
│   ├── pattern_detector.py # Concept extraction agent (PD)
│   └── metacognitive.py    # Session monitoring agent (MA)
├── core/                    # Core services and orchestration
│   ├── orchestrator.py     # Main ExplorationEngine class
│   ├── embeddings.py       # Vector embedding service
│   └── analytics.py        # Session analysis and metrics
├── database/                # Database clients and connections
│   ├── neo4j_client.py     # Knowledge graph operations
│   └── mongodb_client.py   # Session and message storage
├── models/                  # Pydantic data models
│   ├── config.py           # Settings and configuration models
│   └── graph.py            # Graph entities (Node, Edge, Session)
├── tools/                   # Utility scripts and importers
│   ├── import_knowledge_graphs.py
│   └── clear_databases.py
└── ui/                      # Future UI components
    └── __init__.py
```

## Configuration & Documentation

```
docs/
├── external/
│   └── persona.md          # Socratic agent persona definition
└── AZURE_GPT5_SETUP.md    # Azure OpenAI setup guide

knowledge_graphs/           # Domain knowledge and connections
├── domains/                # Individual domain knowledge graphs
│   ├── ai_exploration.json
│   ├── chemistry.json
│   ├── philosophy.json
│   └── [other domains].json
└── connections/            # Cross-domain metaphorical connections
    ├── metaphor_connections.json
    ├── base.json
    └── seed_decoding_discovery.json
```

## Architecture Patterns

**Three-Agent System**
- `ConversationalAgent` (CA): Socratic dialogue using metaphors
- `PatternDetector` (PD): Extracts concepts and relationships from conversations
- `MetacognitiveAgent` (MA): Monitors sessions and adjusts system behavior

**Data Flow Pattern**
```
User Input → CA (response) → PD (extract patterns) → MA (analyze & adjust)
                ↓
Neo4j Graph ← Pattern Storage ← Concept Extraction
```

**Database Separation**
- Neo4j: Knowledge graph (nodes, edges, vectors) - single source of truth
- MongoDB: Session data, message history, analytics - transient operational data

## Code Organization Principles

**Async/Await Throughout**
- All database operations are async
- LLM API calls use async clients
- CLI commands wrap async functions with `asyncio.run()`

**Pydantic Models**
- All data structures use Pydantic for validation
- Settings loaded from environment via `pydantic-settings`
- Graph entities (Node, Edge, Session) have strict typing

**Configuration Hierarchy**
1. Environment variables (`.env` file)
2. Settings class with defaults
3. Specialized config objects (LLMConfig, DatabaseConfig, PersonaConfig)

**Error Handling Strategy**
- Graceful degradation with fallback models
- Rich console output for user-friendly error messages
- Comprehensive logging to `peengine.log`

## File Naming Conventions

- **Modules**: Snake_case (e.g., `pattern_detector.py`)
- **Classes**: PascalCase (e.g., `ConversationalAgent`)
- **Functions/Methods**: Snake_case (e.g., `process_user_input`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_TEMPERATURE`)
- **CLI Commands**: Kebab-case (e.g., `import-graphs`, `start-fresh`)

## Import Patterns

**Relative Imports Within Package**
```python
from ..models.graph import Node, Edge, Session
from ..database.neo4j_client import Neo4jClient
from .conversational import ConversationalAgent
```

**External Dependencies**
```python
import typer
from rich.console import Console
from openai import AsyncAzureOpenAI
from neo4j import GraphDatabase
```

## Testing Structure

```
tests/
├── __init__.py
├── test_embeddings.py      # Vector embedding tests
└── test_models.py          # Pydantic model validation tests
```

**Testing Patterns**
- Use pytest for all tests
- Async test functions with `pytest-asyncio`
- Mock external services (OpenAI, databases) in tests
- Focus on data model validation and core logic