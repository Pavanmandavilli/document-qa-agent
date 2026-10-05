from app.core.config import get_settings
from app.llm.base import LLMProvider
from app.llm.gemini import GeminiProvider
from app.llm.groq import GroqProvider
from app.llm.openrouter import OpenRouterProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "openrouter": OpenRouterProvider,
    "groq": GroqProvider,
}


def get_llm() -> LLMProvider:
    name = get_settings().llm_provider.lower()
    try:
        provider_cls = _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Choose one of: {', '.join(_PROVIDERS)}")
    return provider_cls()
