from app.core.config import get_settings
from app.llm.openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"

    def __init__(self):
        s = get_settings()
        if not s.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set")
        super().__init__(
            url="https://api.groq.com/openai/v1/chat/completions",
            api_key=s.groq_api_key,
            model=s.groq_model,
            timeout=s.llm_timeout,
            stream_idle_timeout=s.stream_idle_timeout,
        )
