from app.core.config import get_settings
from app.llm.openai_compatible import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"

    def __init__(self):
        s = get_settings()
        if not s.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        super().__init__(
            url="https://openrouter.ai/api/v1/chat/completions",
            api_key=s.openrouter_api_key,
            model=s.openrouter_model,
            timeout=s.llm_timeout,
            stream_idle_timeout=s.stream_idle_timeout,
            extra_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Document Q&A Agent",
            },
        )
