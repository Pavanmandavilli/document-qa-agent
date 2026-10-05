import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.api.deps import get_embedding_provider
from app.vector_db.db import dispose_engine, init_db

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)

log_file = Path(os.getenv("LOG_FILE", "logs/app.log"))
log_file.parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    embeddings = await asyncio.to_thread(get_embedding_provider)
    await init_db(embeddings.dimension)
    logger.info("startup complete (embedding dimension=%d)", embeddings.dimension)
    yield
    await dispose_engine()


app = FastAPI(title="Document Q&A Agent Service", version="1.1.0", lifespan=lifespan)
logger.info("app initialized with title=%s version=%s", app.title, app.version)
app.include_router(router)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


@app.get("/")
async def root():
    return {
        "service": "Document Q&A Agent Service",
        "docs": "/docs",
        "health": "/health",
    }
