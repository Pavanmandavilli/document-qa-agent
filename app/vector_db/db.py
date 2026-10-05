from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_connection_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db(dimension: int) -> None:
    dimension = int(dimension) 
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY,
                filename TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """))
        await conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id UUID PRIMARY KEY,
                document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR({dimension}) NOT NULL
            )
        """))
        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
            ON document_chunks USING hnsw (embedding vector_cosine_ops)
        """))


async def dispose_engine() -> None:
    await engine.dispose()
