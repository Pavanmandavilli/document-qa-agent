from functools import lru_cache

from app.agents.agents import AnswerAgent, CheckerAgent, RetrieverAgent, RouterAgent
from app.agents.orchestrator import Orchestrator
from app.core.config import get_settings
from app.embeddings.embeddings import EmbeddingProvider
from app.ingestion.loader import IngestionService
from app.llm.base import LLMProvider
from app.llm.factory import get_llm
from app.retrieval.retriever import PgVectorRetriever, Retriever
from app.utils.progress import ProgressBus


@lru_cache
def get_llm_provider() -> LLMProvider:
    return get_llm()


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return EmbeddingProvider(get_settings().embedding_model)


@lru_cache
def get_retriever() -> Retriever:
    return PgVectorRetriever()


@lru_cache
def get_progress_bus() -> ProgressBus:
    return ProgressBus()


@lru_cache
def get_ingestion_service() -> IngestionService:
    return IngestionService(get_embedding_provider(), get_retriever(), get_settings())


@lru_cache
def get_orchestrator() -> Orchestrator:
    settings = get_settings()
    llm = get_llm_provider()
    return Orchestrator(
        router=RouterAgent(llm, settings),
        retriever=RetrieverAgent(get_embedding_provider(), get_retriever(), settings),
        answerer=AnswerAgent(llm, settings),
        checker=CheckerAgent(llm, settings),
        bus=get_progress_bus(),
        settings=settings,
    )
