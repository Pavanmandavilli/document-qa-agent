import asyncio
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader


@dataclass
class PageText:
    page_number: int
    text: str


class IngestionError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def validate_pdf(pdf_bytes: bytes, max_bytes: int) -> None:
    if not pdf_bytes:
        raise IngestionError("Uploaded PDF is empty")
    if len(pdf_bytes) > max_bytes:
        raise IngestionError(f"PDF is larger than the {max_bytes // (1024 * 1024)} MB limit", 413)
    if not pdf_bytes.startswith(b"%PDF-"):
        raise IngestionError("File is not a valid PDF", 415)


def _extract_pages(pdf_bytes: bytes) -> list[PageText]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PageText(page_number, text))
    return pages


async def extract_pages(pdf_bytes: bytes) -> list[PageText]:
    return await asyncio.to_thread(_extract_pages, pdf_bytes)
