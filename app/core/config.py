from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class Settings(BaseSettings):
    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash-lite"
    openrouter_api_key: str = ""
    openrouter_model: str = "openrouter/free"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    json_mode: bool = False

    database_url: str = ""
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "document_qa"
    database_user: str = "postgres"
    database_password: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    top_k: int = 5
    min_score: float = 0.25
    chunk_size: int = 1000
    chunk_overlap: int = 150
    max_context_chars: int = 18000
    max_upload_mb: int = 20
    documents_dir: Path = Path("docs")

    llm_timeout: float = 60.0
    stream_idle_timeout: float = 30.0
    agent_step_timeout: float = 45.0
    request_timeout: float = 180.0
    max_agent_steps: int = 8
    enable_checker: bool = True
    max_answer_retries: int = 1

    ask_rate_limit: int = 10
    ask_rate_period: float = 60.0

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def database_connection_url(self) -> str | URL:
        if self.database_url:
            return self.database_url

        return URL.create(
            "postgresql+asyncpg",
            username=self.database_user,
            password=self.database_password,
            host=self.database_host,
            port=self.database_port,
            database=self.database_name,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
