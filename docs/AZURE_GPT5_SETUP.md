# Azure OpenAI GPT-5 Setup Guide

This guide covers configuring the Personal Exploration Engine to use Azure OpenAI with GPT-5.

## Prerequisites

1. **Azure OpenAI Resource**: You need an Azure OpenAI resource with GPT-5 access
2. **GPT-5 Deployment**: Create a deployment of your GPT-5 model in Azure OpenAI Studio
3. **API Key**: Get your Azure OpenAI API key from the resource

## Configuration Steps

### 1. Create Azure OpenAI Deployment

1. Go to [Azure OpenAI Studio](https://oai.azure.com/)
2. Navigate to **Deployments** → **Create new deployment**
3. Select your GPT-5 model (e.g., `gpt-5-chat` or `gpt-5`)
4. Give it a deployment name (e.g., `peengine-gpt5`)
5. Note down the deployment name - you'll need this

### 2. Get Your Credentials

From your Azure OpenAI resource page:
- **Endpoint URL**: Found in "Keys and Endpoint" (e.g., `https://your-resource.cognitiveservices.azure.com/`)
- **API Key**: One of the keys from "Keys and Endpoint"
- **API Version**: Use `2024-12-01-preview` 

### 3. Configure Environment

Edit your `.env` file with your Azure OpenAI settings. See `.env.example` for the full list of variables. The key settings are:

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

# Neo4j Configuration (unchanged)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here

# Application Configuration (unchanged)
LOG_LEVEL=INFO
PERSONA_PATH=docs/external/persona.md
KNOWLEDGE_GRAPHS_PATH=knowledge_graphs/
```

### 4. Model-Specific Optimizations

GPT-5 has enhanced capabilities that the system will automatically leverage:

**Enhanced Reasoning**: GPT-5's improved reasoning will benefit:
- Socratic questioning quality (CA)
- Pattern extraction accuracy (PD) 
- Metacognitive analysis depth (MA)

**Better Context Handling**: GPT-5's extended context window allows:
- Longer conversation histories
- More comprehensive session analysis
- Better cross-session knowledge integration

**Improved Metaphor Generation**: GPT-5's enhanced creativity will improve:
- More sophisticated metaphor bridging
- Better domain-crossing analogies
- More nuanced understanding detection

### 5. Persona Adjustments for GPT-5

Consider updating your `docs/external/persona.md` to leverage GPT-5's capabilities:

```markdown
# Enhanced Persona for GPT-5

## Advanced Socratic Techniques
With GPT-5's enhanced reasoning, you can:
- Use more complex logical chains in questioning
- Detect subtle conceptual gaps more accurately
- Generate multi-layered metaphors that span several domains

## Meta-Reasoning
GPT-5 can better understand:
- When the learner is ready for canonical knowledge
- How to adjust questioning depth dynamically
- When to switch between exploration modes

## Improved Falsification
GPT-5 excels at:
- Generating stronger counterexamples
- Identifying edge cases in metaphors
- Creating more rigorous hypothesis testing
```

### 6. Testing the Setup

Run these commands to verify GPT-5 integration:

```bash
# Initialize and test configuration
peengine init

# Start a test session
peengine start "test GPT-5 reasoning"

# Try advanced queries that leverage GPT-5's capabilities
# - Complex multi-domain questions
# - Abstract philosophical concepts
# - Technical topics requiring deep reasoning
```

### 7. Performance Optimization

**Token Management**: GPT-5 has higher token limits, but also higher costs:
- Adjust `OPENAI_MAX_TOKENS` if needed (default: 2000)
- Monitor usage in Azure portal

**Rate Limits**: Check your Azure OpenAI quotas:
- GPT-5 may have different rate limits than GPT-4
- Consider request batching for high-frequency operations

**Cost Optimization**: GPT-5 is more expensive:
- Monitor costs via Azure Cost Management
- Consider using GPT-4 for routine operations and GPT-5 for complex reasoning

## Troubleshooting

### Common Issues

**Deployment Not Found**:
```
Error: The API deployment for this resource does not exist.
```
- Verify `OPENAI_DEPLOYMENT_NAME` matches your Azure deployment exactly
- Check deployment status in Azure OpenAI Studio

**API Version Mismatch**:
```
Error: Invalid API version.
```
- Use `2024-02-15-preview` or later for GPT-5 access
- Check Azure OpenAI documentation for latest API versions

**Authentication Issues**:
```
Error: Access denied or invalid API key.
```
- Verify API key is from the correct Azure resource
- Check that your resource has GPT-5 access enabled

### Debug Mode

Enable detailed logging to diagnose issues:

```env
LOG_LEVEL=DEBUG
```

This will show:
- Full API requests/responses
- Token usage details
- Model selection logic

## Advanced Features

With GPT-5, you can enable advanced features:

### Enhanced Vector Operations
GPT-5's better understanding improves:
- u-vector/c-vector gap analysis
- Metaphor connection strength calculation
- Cross-domain similarity detection

### Improved Session Analysis
The Metacognitive Agent can:
- Detect more subtle learning patterns
- Generate more sophisticated exploration seeds
- Provide deeper session insights

### Better Knowledge Integration
GPT-5 excels at:
- Connecting concepts across long conversation histories
- Maintaining coherent mental models over extended sessions
- Detecting when canonical knowledge should be introduced

---

## Next Steps

Once configured:
1. Test with complex philosophical or technical topics
2. Compare reasoning quality with previous GPT-4 sessions
3. Experiment with deeper exploration sessions (20+ exchanges)
4. Monitor how GPT-5's enhanced capabilities affect learning outcomes

The Personal Exploration Engine is designed to scale with more powerful models - GPT-5's capabilities should significantly enhance the Socratic dialogue quality and metacognitive analysis depth.