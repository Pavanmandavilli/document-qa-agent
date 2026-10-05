import asyncio
import logging
import re
from collections.abc import AsyncIterator
from contextlib import aclosing

from app.core.config import Settings, get_settings
from app.core.decorators import timed
from app.embeddings.embeddings import EmbeddingProvider
from app.llm.base import LLMProvider
from app.prompts.prompts import answer_messages, checker_messages, router_messages
from app.retrieval.retriever import Retriever
from app.schemas.models import Citation, RetrievedChunk, RouterDecision, Verdict
from app.utils.structured import parse_json_model

logger = logging.getLogger(__name__)

REFUSAL = "I don't know."
_SMALL_TALK = {"hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening"}
_CITATION_MARKER = re.compile(r"\[([^\[\]]+?),\s*(\d+)\]")


def is_refusal(answer: str) -> bool:
    normalized = answer.strip().lower().replace("\u2019", "'")
    return normalized.startswith(("i don't know", "i do not know"))


def build_context(chunks: list[RetrievedChunk], max_chars: int) -> str:
    parts: list[str] = []
    used = 0
    for i, chunk in enumerate(chunks, 1):
        block = f"[SOURCE {i}] Document: {chunk.document} | Page: {chunk.page}\n{chunk.content}"
        if parts and used + len(block) > max_chars:
            break
        parts.append(block[:max_chars])
        used += len(block)
    return "\n\n".join(parts)


def citations_from_text(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    available = {(c.document, c.page) for c in chunks}
    found: list[Citation] = []
    for doc, page in _CITATION_MARKER.findall(answer):
        key = (doc.strip(), int(page))
        if key in available and Citation(document=key[0], page=key[1]) not in found:
            found.append(Citation(document=key[0], page=key[1]))
    if not found and chunks:
        found.append(Citation(document=chunks[0].document, page=chunks[0].page))
    return found


class RouterAgent:
    def __init__(self, llm: LLMProvider, settings: Settings | None = None):
        self.llm = llm
        self.settings = settings or get_settings()

    @timed
    async def route(self, question: str) -> RouterDecision:
        response_format = {"type": "json_object"} if self.settings.json_mode else None
        try:
            raw = await self.llm.generate(
                router_messages(question), response_format=response_format, temperature=0
            )
            return parse_json_model(raw, RouterDecision)
        except Exception as exc:
            logger.warning("router failed (%r); using keyword fallback", exc)
            return self._fallback_route(question)

    @staticmethod
    def _fallback_route(question: str) -> RouterDecision:
        if question.lower().strip(" !.?") in _SMALL_TALK:
            return RouterDecision(route="small_talk", reason="Simple conversational message")
        return RouterDecision(route="document_qa", reason="Fallback to document retrieval")


class RetrieverAgent:
    def __init__(
        self, embeddings: EmbeddingProvider, retriever: Retriever, settings: Settings | None = None
    ):
        self.embeddings = embeddings
        self.retriever = retriever
        self.settings = settings or get_settings()

    @timed
    async def retrieve(self, question: str) -> list[RetrievedChunk]:
        query_embedding = await self.embeddings.embed_one(question)
        candidates = await self.retriever.search(
            question, query_embedding, self.settings.top_k
        )
        relevant = [
            chunk
            for chunk in candidates
            if (
                (chunk.vector_score is not None and chunk.vector_score >= self.settings.min_score)
                or (chunk.bm25_score is not None and chunk.bm25_score > 0)
                or (chunk.vector_score is None and chunk.score >= self.settings.min_score)
            )
        ]
        logger.info(
            "retrieved candidates=%d relevant=%d best_score=%s",
            len(candidates),
            len(relevant),
            f"{candidates[0].score:.3f}" if candidates else "n/a",
        )
        return relevant


class AnswerAgent:
    def __init__(self, llm: LLMProvider, settings: Settings | None = None):
        self.llm = llm
        self.settings = settings or get_settings()

    async def stream(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        retry_reason: str | None = None,
    ) -> AsyncIterator[str]:
        context = build_context(chunks, self.settings.max_context_chars)
        messages = answer_messages(question, context, retry_reason)

        async with aclosing(self.llm.stream(messages, temperature=0)) as tokens:
            while True:
                try:
                    token = await asyncio.wait_for(
                        anext(tokens), timeout=self.settings.stream_idle_timeout
                    )
                except StopAsyncIteration:
                    return
                yield token


class CheckerAgent:
    def __init__(self, llm: LLMProvider, settings: Settings | None = None):
        self.llm = llm
        self.settings = settings or get_settings()

    @timed
    async def verify(self, question: str, answer: str, chunks: list[RetrievedChunk]) -> Verdict:
        context = build_context(chunks, self.settings.max_context_chars)
        response_format = {"type": "json_object"} if self.settings.json_mode else None
        raw = await self.llm.generate(
            checker_messages(question, answer, context),
            response_format=response_format,
            temperature=0,
        )
        verdict = parse_json_model(raw, Verdict)

        available = {(c.document, c.page) for c in chunks}
        valid: list[Citation] = []
        for citation in verdict.citations:
            if (citation.document, citation.page) in available and citation not in valid:
                valid.append(citation)

        if verdict.supported and not valid:
            valid = citations_from_text(answer, chunks)
        verdict.citations = valid if verdict.supported else []
        return verdict
