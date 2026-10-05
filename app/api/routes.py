import asyncio
import json
import logging
import os

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse, StreamingResponse

from app.agents.orchestrator import Orchestrator
from app.api.deps import get_ingestion_service, get_orchestrator, get_progress_bus, get_retriever
from app.core.config import get_settings
from app.core.decorators import rate_limit, timed
from app.ingestion.loader import IngestionService
from app.ingestion.parser import IngestionError
from app.retrieval.retriever import Retriever
from app.schemas.models import AskRequest, UploadResponse
from app.utils.progress import ProgressBus

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/documents", response_model=UploadResponse, status_code=201)
@timed
async def upload_document(
    file: UploadFile = File(...),
    ingestion: IngestionService = Depends(get_ingestion_service),
):
    filename = os.path.basename(file.filename or "")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported")

    limit = settings.max_upload_mb * 1024 * 1024
    data = await file.read(limit + 1)

    try:
        result = await ingestion.ingest(filename, data)
    except IngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return UploadResponse(
        document_id=str(result.document_id),
        filename=result.filename,
        pages=result.pages,
        chunks=result.chunks,
    )


@router.post("/ask")
@rate_limit(max_calls=settings.ask_rate_limit, period=settings.ask_rate_period)
@timed
async def ask(
    request: Request,
    body: AskRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    async def event_stream():
        async for event, data in orchestrator.run(body.question):
            yield sse(event, data)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/health")
async def health(retriever: Retriever = Depends(get_retriever)):
    ok = await retriever.healthy()
    return JSONResponse(
        {
            "status": "ok" if ok else "degraded",
            "llm": settings.llm_provider,
            "vector_store": type(retriever).__name__,
        },
        status_code=200 if ok else 503,
    )


@router.websocket("/ws/progress")
async def progress_ws(websocket: WebSocket, bus: ProgressBus = Depends(get_progress_bus)):
    await websocket.accept()
    queue = bus.subscribe()

    async def sender() -> None:
        while True:
            await websocket.send_json(await queue.get())

    async def watch_disconnect() -> None:
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            return

    tasks = [asyncio.create_task(sender()), asyncio.create_task(watch_disconnect())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        bus.unsubscribe(queue)
