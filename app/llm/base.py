from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class LLMError(Exception):
    pass

class LLMProvider(ABC):
    name: str = "llm"

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        ...
