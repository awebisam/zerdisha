"""Optional live LLM test hitting Azure Responses API.

Skipped unless AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT_NAME are set.
"""

import os
import json
import pytest

from peengine.core.llm_client import AzureLLMClient


required_env = [
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION",
]


def has_env():
    return all(os.getenv(k) for k in required_env)


pytestmark = pytest.mark.skipif(
    not has_env(), reason="Live LLM env vars not set; skipping"
)


@pytest.mark.asyncio
async def test_responses_api_json_schema_live():
    client = AzureLLMClient(
        api_key=os.environ["AZURE_OPENAI_KEY"],
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    )

    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string"},
            "n": {"type": "number"}
        },
        "required": ["a", "n"]
    }

    content = await client.json_response(
        "Return a tiny JSON with keys a (string) and n (number).",
        schema=schema,
        temperature=0,
        max_tokens=64,
    )

    data = json.loads(content)
    assert isinstance(data["a"], str)
    assert isinstance(data["n"], (int, float))
