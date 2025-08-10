# Personal Exploration Engine (PEEngine)

## Not applicable for anyone except @awebisam
## Modeled around my cognitive structure only

A terminal-based interactive learning tool that serves as your *personal cognitive exoskeleton* for exploring topics, mapping conceptual connections, and refining mental models through AI-guided Socratic dialogue.

## 🧠 Core Philosophy

- **Exploration over Answers**: AI guides you to discover insights rather than providing direct answers
- **Metaphorical Reasoning**: Uses metaphors as bridges between domains and concepts
- **Graph-based Memory**: Every concept and connection is stored in a Neo4j knowledge graph
- **Metacognitive Awareness**: System monitors and adjusts its own behavior based on learning patterns

## 🏗️ Architecture

The system integrates three AI agents:

- **Conversational Agent (CA)** — Your Socratic guide using metaphors
- **Pattern Detector (PD)** — Maps concepts and relationships to Neo4j graph
- **Metacognitive Agent (MA)** — Monitors sessions and adjusts system parameters

## 🚀 Quick Start

### 1. Setup

```bash
# Clone and install
git clone <repo-url>
cd zerdisha
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your OpenAI API key and Neo4j credentials

# Initialize
peengine init
```

### 2. Start Exploring

```bash
# Start a new exploration session
peengine start "quantum mechanics"

# Interactive session with commands:
# /map     - Show concept map
# /seed    - Get exploration seed from MA
# /gapcheck - Check understanding gaps
# /end     - Save and close session
```

### 3. Review Past Sessions

```bash
# Review by date or session ID
peengine review 2025-08-09
peengine status
```

## 📋 Requirements

- Python 3.8+
- Neo4j database (local or cloud)
- OpenAI API key (or Azure OpenAI)
  - **GPT-5 Support**: Optimized for Azure OpenAI GPT-5 models
  - **GPT-4 Compatible**: Works with GPT-4, GPT-4 Turbo, and GPT-3.5

## 🔧 Configuration

Key settings in `.env`:

```env
# OpenAI (Standard API)
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4
OPENAI_API_TYPE=openai

# Azure OpenAI (for GPT-5)
# OPENAI_API_KEY=your_azure_key
# OPENAI_BASE_URL=https://your-resource.openai.azure.com/
# OPENAI_MODEL=gpt-5-turbo
# OPENAI_API_TYPE=azure
# OPENAI_DEPLOYMENT_NAME=your-gpt5-deployment

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Paths
PERSONA_PATH=docs/external/persona.md
KNOWLEDGE_GRAPHS_PATH=knowledge_graphs/
```

**For GPT-5 setup**: See [Azure GPT-5 Setup Guide](docs/AZURE_GPT5_SETUP.md)

## 🎯 Key Features

### Socratic Dialogue
- AI never gives direct answers
- Uses metaphors as primary communication method
- Adapts metaphors if learner doesn't understand
- Challenges assumptions through questioning

### Knowledge Graph
- **u-vectors**: Your personal, metaphorical understanding
- **c-vectors**: Canonical academic knowledge
- **Metaphor connections**: Cross-domain bridges
- **Session tracking**: Every exploration is preserved

### Metacognitive Monitoring
- Detects metaphor lock-in, topic drift, stagnation
- Suggests new exploration seeds
- Adjusts conversational parameters in real-time
- Generates insights about learning patterns

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

Based on your `philosophy.json`, the system embodies:

- **Non-dual awareness**: Everything connects to everything
- **Process over product**: Focus on exploration rather than conclusions  
- **Embodied cognition**: Mental models are tools, not truth
- **Meta-cognitive learning**: Learning how to learn

## ⚠️ Current Status

This is an experimental learning tool built according to the PRD. It implements:

- ✅ Full Typer CLI with interactive sessions
- ✅ Neo4j graph storage with vector support
- ✅ Three-agent architecture (CA/PD/MA)
- ✅ Session management and review
- ✅ Vector embeddings (u-vectors/c-vectors)
- ✅ Rich TUI with conversation flow

## 🔮 Future Enhancements

- Graph visualization in TUI
- Advanced vector similarity search
- Integration with existing knowledge graphs
- Export capabilities for artifacts
- Multi-session learning trajectories

## 🤝 Contributing

This is primarily a personal learning experiment. If you're interested in the approach:

1. Try the system with your own explorations
2. Document what works/doesn't work
3. Submit insights rather than just code fixes

## 📄 License

MIT License - See LICENSE file for details.

---

*"The best way to understand a system is to try to change it."* - Kurt Lewin