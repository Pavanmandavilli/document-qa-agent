from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
import re
from math import log
from uuid import UUID, uuid4

from sqlalchemy import text

from app.schemas.models import EmbeddedChunk, RetrievedChunk
from app.vector_db.db import SessionLocal


class Retriever(ABC):
    @abstractmethod
    async def add_document(
        self, document_id: UUID, filename: str, chunks: Sequence[EmbeddedChunk]
    ) -> None:
        ...

    @abstractmethod
    async def search(
        self, query: str, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        ...

    async def healthy(self) -> bool:
        return True


def _vector_literal(embedding: Sequence[float]) -> str:
    return "[" + ",".join(map(str, embedding)) + "]"


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _bm25_rank(
    query: str, documents: list[RetrievedChunk], k1: float = 1.5, b: float = 0.75
) -> list[tuple[RetrievedChunk, float]]:
    query_terms = set(_tokens(query))
    if not query_terms or not documents:
        return []

    tokenized = [Counter(_tokens(document.content)) for document in documents]
    document_count = len(documents)
    document_lengths = [sum(counts.values()) for counts in tokenized]
    average_length = sum(document_lengths) / document_count
    if average_length == 0:
        return []
    document_frequency = {
        term: sum(term in counts for counts in tokenized)
        for term in query_terms
    }

    scored: list[tuple[float, RetrievedChunk]] = []
    for document, counts, length in zip(
        documents, tokenized, document_lengths, strict=True
    ):
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if not frequency:
                continue
            frequency_in_docs = document_frequency[term]
            inverse_document_frequency = log(
                1 + (document_count - frequency_in_docs + 0.5) / (frequency_in_docs + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * length / average_length
            )
            score += (
                inverse_document_frequency
                * frequency
                * (k1 + 1)
                / denominator
            )
        if score > 0:
            scored.append((score, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [(document, score) for score, document in scored]


def _fuse_rankings(
    vector_results: list[RetrievedChunk],
    bm25_results: list[tuple[RetrievedChunk, float]],
    top_k: int,
) -> list[RetrievedChunk]:
    reciprocal_rank_constant = 60
    max_score = 2 / (reciprocal_rank_constant + 1)
    by_key: dict[tuple[UUID, int, int], RetrievedChunk] = {}
    scores: dict[tuple[UUID, int, int], float] = {}
    vector_scores: dict[tuple[UUID, int, int], float] = {}
    bm25_scores: dict[tuple[UUID, int, int], float] = {}

    for rank, chunk in enumerate(vector_results, start=1):
        key = (chunk.document_id, chunk.page, chunk.chunk_index)
        by_key[key] = chunk
        vector_scores[key] = chunk.score
        scores[key] = scores.get(key, 0.0) + (
            1 / (reciprocal_rank_constant + rank)
        )

    for rank, (chunk, bm25_score) in enumerate(bm25_results, start=1):
        key = (chunk.document_id, chunk.page, chunk.chunk_index)
        by_key[key] = chunk
        bm25_scores[key] = bm25_score
        scores[key] = scores.get(key, 0.0) + (
            1 / (reciprocal_rank_constant + rank)
        )

    ranked_keys = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return [
        by_key[key].model_copy(
            update={
                "score": scores[key] / max_score,
                "vector_score": vector_scores.get(key),
                "bm25_score": bm25_scores.get(key, 0.0),
            }
        )
        for key in ranked_keys
    ]


class PgVectorRetriever(Retriever):
    async def add_document(self, document_id, filename, chunks):
        rows = [
            {
                "id": uuid4(),
                "document_id": document_id,
                "page": c.page,
                "index": c.index,
                "content": c.text,
                "embedding": _vector_literal(c.embedding),
            }
            for c in chunks
        ]

        async with SessionLocal() as session, session.begin():
            await session.execute(
                text("INSERT INTO documents (id, filename) VALUES (:id, :filename)"),
                {"id": document_id, "filename": filename},
            )
            if rows:
                await session.execute(
                    text("""
                        INSERT INTO document_chunks
                            (id, document_id, page_number, chunk_index, content, embedding)
                        VALUES
                            (:id, :document_id, :page, :index, :content,
                             CAST(:embedding AS vector))
                    """),
                    rows,
                )

    async def search(self, query: str, query_embedding: list[float], top_k: int):
        candidate_k = max(top_k * 2, top_k)
        async with SessionLocal() as session:
            vector_result = await session.execute(
                text("""
                    SELECT dc.document_id, d.filename, dc.page_number, dc.chunk_index,
                           dc.content,
                           1 - (dc.embedding <=> CAST(:embedding AS vector)) AS score
                    FROM document_chunks dc
                    JOIN documents d ON d.id = dc.document_id
                    ORDER BY dc.embedding <=> CAST(:embedding AS vector)
                    LIMIT :candidate_k
                """),
                {"embedding": _vector_literal(query_embedding), "candidate_k": candidate_k},
            )
            corpus_result = await session.execute(
                text("""
                    SELECT dc.document_id, d.filename, dc.page_number, dc.chunk_index,
                           dc.content
                    FROM document_chunks dc
                    JOIN documents d ON d.id = dc.document_id
                """)
            )
            vector_rows = vector_result.mappings().all()
            corpus_rows = corpus_result.mappings().all()

        vector_chunks = [
            RetrievedChunk(
                document_id=row["document_id"],
                document=row["filename"],
                page=row["page_number"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                score=float(row["score"]),
                vector_score=float(row["score"]),
            )
            for row in vector_rows
        ]
        corpus_chunks = [
            RetrievedChunk(
                document_id=row["document_id"],
                document=row["filename"],
                page=row["page_number"],
                chunk_index=row["chunk_index"],
                content=row["content"],
                score=0.0,
            )
            for row in corpus_rows
        ]
        return _fuse_rankings(
            vector_chunks,
            _bm25_rank(query, corpus_chunks)[:candidate_k],
            top_k,
        )

    async def healthy(self) -> bool:
        try:
            async with SessionLocal() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


class InMemoryRetriever(Retriever):
    def __init__(self) -> None:
        self._items: list[tuple[UUID, str, EmbeddedChunk]] = []

    async def add_document(self, document_id, filename, chunks):
        self._items.extend((document_id, filename, c) for c in chunks)

    async def search(self, query: str, query_embedding: list[float], top_k: int):
        vector_chunks = []
        all_chunks = []
        for document_id, filename, chunk in self._items:
            score = sum(a * b for a, b in zip(query_embedding, chunk.embedding, strict=True))
            retrieved = RetrievedChunk(
                document_id=document_id,
                document=filename,
                page=chunk.page,
                chunk_index=chunk.index,
                content=chunk.text,
                score=score,
                vector_score=score,
            )
            vector_chunks.append(retrieved)
            all_chunks.append(retrieved)
        vector_chunks.sort(key=lambda item: item.score, reverse=True)

        candidate_k = max(top_k * 2, top_k)
        return _fuse_rankings(
            vector_chunks[:candidate_k],
            _bm25_rank(query, all_chunks)[:candidate_k],
            top_k,
        )
