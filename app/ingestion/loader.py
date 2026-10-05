import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from app.chunking.chunker import batched, iter_document_chunks
from app.core.config import Settings
from app.embeddings.embeddings import EmbeddingProvider
from app.ingestion.parser import IngestionError, extract_pages, validate_pdf
from app.retrieval.retriever import Retriever
from app.schemas.models import EmbeddedChunk

logger = logging.getLogger(__name__)
EMBED_BATCH_SIZE = 64


@dataclass
class IngestResult:
    document_id: UUID
    filename: str
    pages: int
    chunks: int


class IngestionService:
    def __init__(self, embeddings: EmbeddingProvider, retriever: Retriever, settings: Settings):
        self.embeddings = embeddings
        self.retriever = retriever
        self.settings = settings

    async def ingest(self, filename: str, pdf_bytes: bytes) -> IngestResult:
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        validate_pdf(pdf_bytes, max_bytes)

        try:
            pages = await extract_pages(pdf_bytes)
        except Exception as exc:
            logger.exception("PDF extraction failed")
            raise IngestionError("Could not read PDF") from exc

        if not pages:
            raise IngestionError(
                "PDF contains no extractable text (scanned PDFs need OCR, which is not supported)",
                422,
            )

        chunk_stream = iter_document_chunks(
            pages, self.settings.chunk_size, self.settings.chunk_overlap
        )
        embedded: list[EmbeddedChunk] = []
        for batch in batched(chunk_stream, EMBED_BATCH_SIZE):
            vectors = await self.embeddings.embed([chunk.text for chunk in batch])
            embedded.extend(
                EmbeddedChunk(
                    page=chunk.page,
                    index=chunk.index,
                    text=chunk.text,
                    embedding=vector,
                )
                for chunk, vector in zip(batch, vectors, strict=True)
            )

        document_id = uuid4()
        safe_filename = Path(filename.replace("\\", "/")).name
        source_path = self.settings.documents_dir / f"{document_id.hex}_{safe_filename}"
        try:
            await asyncio.to_thread(self._store_pdf, source_path, pdf_bytes)
        except OSError as exc:
            logger.exception("PDF storage failed filename=%s", filename)
            raise IngestionError("Could not store uploaded PDF", 500) from exc

        try:
            await self.retriever.add_document(document_id, filename, embedded)
        except Exception:
            try:
                await asyncio.to_thread(source_path.unlink, missing_ok=True)
            except OSError:
                logger.exception("Could not remove stored PDF after ingestion failure")
            raise

        logger.info(
            "ingested filename=%s pages=%d chunks=%d",
            filename,
            len(pages),
            len(embedded),
        )
        return IngestResult(document_id, filename, len(pages), len(embedded))

    @staticmethod
    def _store_pdf(path: Path, pdf_bytes: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf_bytes)
