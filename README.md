# Zerdisha

## Not applicable for anyone except @awebisam
## Modeled around my cognitive structure only

A terminal-based interactive learning tool that serves as my *personal cognitive exoskeleton* for exploring topics, mapping conceptual connections, and refining mental models through AI-guided Socratic dialogue.

## 🧠 Core Philosophy

- **Exploration over Answers**: AI guides me to discover insights rather than providing direct answers
- **Metaphorical Reasoning**: Uses metaphors as bridges between domains and concepts
- **Graph-based Memory**: Every concept and connection is stored in a Neo4j knowledge graph
- **Metacognitive Awareness**: System monitors and adjusts its own behavior based on my learning patterns

## 🏗️ Architecture

The system integrates three AI agents:

- **Conversational Agent (CA)** — My Socratic guide using metaphors
- **Pattern Detector (PD)** — Maps concepts and relationships to Neo4j graph
- **Metacognitive Agent (MA)** — Monitors sessions and adjusts system parameters

## 🚀 Quick Start

### 1. Setup

**Option A: Docker (Recommended)**
```bash
# Quick setup with Docker
./setup-docker.sh

# Edit .env with my Azure OpenAI credentials
# Then test
zerdisha init
```

**Option B: Manual Setup**
```bash
# Install dependencies
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with my Azure OpenAI and database credentials

# Start databases manually (Neo4j + MongoDB)
# Initialize
zerdisha init
```

### 2. Start Exploring

```bash
# Start a new exploration session
zerdisha start "quantum mechanics"

# Interactive session with commands:
# /map     - Show concept map
# /seed    - Get exploration seed from MA
# /gapcheck - Check understanding gaps
# /end     - Save and close session
```

### 3. Review Past Sessions

```bash
# Review by date or session ID
zerdisha review 2025-08-09
zerdisha status
```

## 📋 Requirements

- Python 3.8+
- Docker & Docker Compose (for easy setup)
- Azure OpenAI with **GPT-5** deployment (gpt-5-chat)
- Neo4j + MongoDB (automatically handled by Docker)

## 🔧 Configuration

Key settings in `.env`:

```env
# Azure OpenAI Configuration (Primary)
AZURE_OPENAI_KEY=your_azure_openai_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5-chat
AZURE_OPENAI_MODEL_NAME=gpt-5-chat
AZURE_OPENAI_API_VERSION=2024-12-01-preview

# Model Selection
PRIMARY_MODEL=gpt-5-chat
FALLBACK_MODEL=gpt-4.1-mini
PATTERN_MODEL=gpt-4.1-mini

# Database Configuration (handled by Docker)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=knowledge123

MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=socratic_lab
```

## 🎯 Key Features

### Socratic Dialogue
- AI never gives direct answers
- Uses metaphors as primary communication method
- Adapts metaphors if I don't understand
- Challenges assumptions through questioning

### Knowledge Graph
- **u-vectors**: My personal, metaphorical understanding
- **c-vectors**: Canonical academic knowledge
- **Metaphor connections**: Cross-domain bridges
- **Session tracking**: Every exploration is preserved

### Metacognitive Monitoring
- Detects metaphor lock-in, topic drift, stagnation
- Suggests new exploration seeds
- Adjusts conversational parameters in real-time
- Generates insights about my learning patterns

## 📊 Data Models

### Core Entities
- **Nodes**: Concepts, metaphors, domains, sessions
- **Edges**: Metaphorical, canonical, exploratory relationships
- **Vectors**: u-vectors (user understanding) and c-vectors (canonical knowledge)
- **Sessions**: Contiguous exploration with full context

### Session Flow
```
User Input → CA (Socratic response) → PD (extract patterns) → MA (analyze & adjust)
                     ↓
Neo4j Graph ← Pattern Storage ← Concept Extraction
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=peengine
```

## 🏛️ Philosophical Foundation

Based on my `philosophy.json`, the system embodies:

- **Non-dual awareness**: Everything connects to everything
- **Process over product**: Focus on exploration rather than conclusions  
- **Embodied cognition**: Mental models are tools, not truth
- **Meta-cognitive learning**: Learning how to learn

## ⚠️ Current Status

This is an experimental learning tool built according to the PRD. The core feature set is now solidified and includes:

- ✅ Full Typer CLI with interactive sessions
- ✅ Neo4j graph storage with vector support
- ✅ Three-agent architecture (CA/PD/MA) with active metacognitive influence
- ✅ Session management and review
- ✅ Vector embeddings (u-vectors/c-vectors) for gap analysis (`/gapcheck`)
- ✅ Enhanced session mapping with relationship descriptions (`/map`)
- ✅ Rich TUI with conversation flow

## 🔮 Future Enhancements

- Graph visualization in TUI
- User-facing vector similarity search command
- Automated MA influence loop (proactive adjustments)
- Multi-session learning trajectory analysis
- Export capabilities for artifacts

## 🤝 Contributing

This is primarily a personal learning experiment. If interested in the approach:

1. Try the system with different explorations
2. Document what works/doesn't work
3. Submit insights rather than just code fixes

## 📄 License

MIT License - See LICENSE file for details.

## 🧰 Developer guide

If you're working on the codebase, see `.github/DEVELOPER_HELP.md` for a quick repo map, environment variables, CLI workflow, and troubleshooting tips.

---

*"The best way to understand a system is to try to change it."* - Kurt Lewin