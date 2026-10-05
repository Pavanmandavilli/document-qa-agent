import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.decorators import retry_with_backoff, timed
from app.llm.base import LLMError, LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    name = "openai-compatible"

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model: str,
        timeout: float,
        stream_idle_timeout: float,
        extra_headers: dict[str, str] | None = None,
    ):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.timeout = httpx.Timeout(timeout, connect=10.0)
        self.stream_timeout = httpx.Timeout(10.0, read=stream_idle_timeout)
        self.extra_headers = extra_headers or {}

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    @retry_with_backoff(attempts=3)
    @timed
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()

        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response from {self.name}: {str(data)[:200]}") from exc

    @retry_with_backoff(attempts=3)
    @timed
    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=self.stream_timeout) as client:
            async with client.stream(
                "POST", self.url, headers=self.headers, json=payload
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    choices = event.get("choices") or []
                    if choices:
                        content = (choices[0].get("delta") or {}).get("content")
                        if content:
                            yield content
