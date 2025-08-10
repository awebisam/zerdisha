# Technology Stack & Build System

## Core Technologies

**Language & Framework**
- Python 3.8+ with async/await patterns
- Typer CLI framework for command-line interface
- Rich library for terminal UI rendering
- Pydantic for data modeling and validation

**AI & LLM Integration**
- Azure OpenAI (primary) with GPT-5 deployment
- Azure AI Foundry (fallback)
- OpenAI Python client with async support
- Model hierarchy: gpt-5-chat (primary) → gpt-4.1-mini (fallback) → gpt-4.1-mini (pattern detection)

**Databases**
- Neo4j 5.15+ for knowledge graph storage with vector support
- MongoDB for session data and message history
- Both databases run via Docker containers

**Key Dependencies**
- `typer[all]>=0.9.0` - CLI framework
- `rich>=13.0.0` - Terminal UI
- `neo4j>=5.0.0` - Graph database client
- `openai>=1.0.0` - LLM integration
- `pydantic>=2.0.0` - Data validation
- `pymongo>=4.0.0` - MongoDB client
- `textual>=0.41.0` - Advanced TUI components

## Build System

**Package Management**
- Uses `pyproject.toml` with setuptools build backend
- Editable installation: `pip install -e .`
- Entry point: `peengine` command via CLI module

**Development Setup**
```bash
# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"
```

## Common Commands

**Environment Setup**
```bash
# Copy environment template
cp .env.example .env
# Edit .env with Azure OpenAI credentials

# Docker setup (recommended)
./setup-docker.sh

# Manual database setup
docker-compose up -d neo4j mongodb
```

**Application Commands**
```bash
# Initialize and check configuration
peengine init

# Start exploration session
peengine start "topic name"

# Review past sessions
peengine review 2025-08-09
peengine status

# Import knowledge graphs
peengine import-graphs

# Clear databases (destructive)
peengine start-fresh
```

**Development & Testing**
```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=peengine

# Code formatting
black peengine/
flake8 peengine/

# Type checking
mypy peengine/
```

**Database Management**
```bash
# Access Neo4j browser
open http://localhost:7474

# Access MongoDB Express
open http://localhost:8081

# Check database health
docker-compose ps
```

## Configuration

**Environment Variables** (`.env` file)
- Azure OpenAI: `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`
- Database: `NEO4J_PASSWORD`, `MONGODB_URI`
- Models: `PRIMARY_MODEL`, `FALLBACK_MODEL`, `PATTERN_MODEL`

**Key Settings**
- Persona file: `docs/external/persona.md`
- Knowledge graphs: `knowledge_graphs/` directory
- Logging: Configurable via `LOG_LEVEL` environment variable