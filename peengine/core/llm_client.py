"""Shared LLM client that prefers Azure Responses API with JSON schema and falls back to chat."""

import logging
from typing import Any, Dict, Optional

from openai import AsyncAzureOpenAI

logger = logging.getLogger(__name__)


class AzureLLMClient:
    def __init__(self, api_key: str, endpoint: str, api_version: str, deployment: str):
        self.deployment = deployment
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )

    async def json_response(self, prompt: str, schema: Optional[Dict[str, Any]] = None, temperature: float = 0.3, max_tokens: int = 1000) -> str:
        """Attempt to get JSON from Responses API; fall back to chat JSON mode; then plain chat."""
        # 1) Responses API with JSON schema
        if schema:
            try:
                resp = await self.client.responses.create(
                    model=self.deployment,
                    input=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "schema", "schema": schema}},
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                # Azure SDK returns output in a content array; extract text
                if resp and getattr(resp, "output", None):
                    parts = getattr(resp.output[0], "content", None) or []
                    for part in parts:
                        if getattr(part, "type", "") == "output_text":
                            return getattr(part, "text", "")
            except Exception as e:
                logger.info(
                    f"Responses API with schema failed, will fall back: {e}")

        # 2) Chat JSON mode
        try:
            chat = await self.client.chat.completions.create(
                model=self.deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            content = (chat.choices[0].message.content or "").strip()
            if content:
                return content
        except Exception as e:
            logger.info(f"Chat JSON mode failed, will fall back: {e}")

        # 3) Plain chat
        chat = await self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (chat.choices[0].message.content or "").strip()
