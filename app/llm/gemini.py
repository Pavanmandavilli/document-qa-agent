import asyncio
from collections.abc import AsyncIterator
from typing import Any

from google import genai

from app.core.config import get_settings
from app.core.decorators import retry_with_backoff, timed
from app.llm.base import LLMError, LLMProvider


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self):
        settings = get_settings()
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_model
        self.timeout = settings.llm_timeout
        self.stream_idle_timeout = settings.stream_idle_timeout

    @staticmethod
    def _convert_messages(messages: list[dict[str, str]]) -> tuple[str | None, str]:
        system_instruction: list[str] = []
        conversation: list[str] = []

        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                system_instruction.append(content)
                continue

            if role == "user":
                conversation.append(f"User: {content}")
            elif role == "assistant":
                conversation.append(f"Assistant: {content}")
            else:
                raise LLMError(f"Unsupported Gemini message role: {role}")

        if not conversation:
            raise LLMError("Gemini request must contain a user or assistant message")

        return "\n".join(system_instruction) or None, "\n\n".join(conversation)

    def _generate_sync(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None,
        temperature: float,
        timeout: float,
    ) -> str:
        system_instruction, prompt = self._convert_messages(messages)
        with genai.Client(api_key=self.api_key) as client:
            interaction = client.interactions.create(
                model=self.model,
                input=prompt,
                system_instruction=system_instruction,
                generation_config={"temperature": temperature},
                response_mime_type="application/json" if response_format else None,
                timeout=timeout,
            )
        return interaction.output_text or ""

    @retry_with_backoff(attempts=3)
    @timed
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        return await asyncio.to_thread(
            self._generate_sync,
            messages,
            response_format,
            temperature,
            self.timeout,
        )

    @retry_with_backoff(attempts=3)
    @timed
    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        answer = await asyncio.to_thread(
            self._generate_sync,
            messages,
            None,
            temperature,
            self.timeout,
        )
        if answer:
            yield answer
