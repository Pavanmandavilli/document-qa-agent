from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TypeVar
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.parser import PageText
from app.schemas.models import ChunkRecord

T = TypeVar("T")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> Iterator[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    yield from splitter.split_text(text)


def iter_document_chunks(
    pages: Iterable[PageText], chunk_size: int, overlap: int
) -> Iterator[ChunkRecord]:
    for page in pages:
        for index, text in enumerate(chunk_text(page.text, chunk_size, overlap)):
            yield ChunkRecord(page=page.page_number, index=index, text=text)


def batched(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    if size <= 0:
        raise ValueError("size must be positive")
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch
