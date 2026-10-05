from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class Citation(BaseModel):
    document: str
    page: int = Field(ge=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int


class RouterDecision(BaseModel):
    route: Literal["document_qa", "small_talk", "out_of_scope"]
    reason: str = ""
    reply: str | None = None


class Verdict(BaseModel):
    supported: bool
    reason: str = ""
    citations: list[Citation] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    document_id: UUID
    document: str
    page: int
    chunk_index: int
    content: str
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None


@dataclass(frozen=True)
class ChunkRecord:
    page: int
    index: int
    text: str


@dataclass(frozen=True)
class EmbeddedChunk:
    page: int
    index: int
    text: str
    embedding: list[float]
