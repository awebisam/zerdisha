# Baseline update (LLM + prompts + tests)

This documents the current baseline for LLM usage, prompt templates, agents, and a live test.

- Shared LLM client: `peengine/core/llm_client.py`
  - Responses-first: tries Azure Responses API with JSON Schema, then Chat JSON mode, then plain Chat.
  - Central place to control temperature, max tokens, and consistent parsing.

- Prompt library: `peengine/core/prompts.py`
  - Centralized prompt strings for Metacognitive Agent and Conversational Agent persona synthesis.
  - Prevents drift and makes it easy to iterate on the baseline prompts.

- Agents
  - ConversationalAgent (`peengine/agents/conversational.py`)
    - Adds request timeouts and uses the shared client for persona synthesis with JSON Schema.
    - Keeps existing behavior for general chat and fallback client, plus robust bullet formatting for persona instructions.
  - MetacognitiveAgent (`peengine/agents/metacognitive.py`)
    - Uses centralized prompt templates.
    - Safer template formatting helper to avoid `{}`/format collisions.

- PatternDetector
  - Already on the shared client with schema + strong JSON parsing fallbacks.

- Live LLM test: `tests/test_live_llm.py`
  - Skipped by default. To enable:
    - Set env vars: `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT_NAME`, `AZURE_OPENAI_API_VERSION`.
    - Run the test suite; it verifies a small JSON-schema constrained response.

Notes
- All existing tests pass locally; live test is optional and environment-gated.
- Next iterations can add JSON Schema to more CA/MA flows and expand prompt coverage.
