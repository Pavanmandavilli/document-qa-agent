import asyncio
from sentence_transformers import SentenceTransformer

class EmbeddingProvider:

    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = await asyncio.to_thread(
            self.model.encode,
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]
