"""Optional OpenAI-compatible embedding transport. No LangChain dependency."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from .models import EmbeddingSpec


class OpenAIEmbedder:
    def __init__(
        self,
        *,
        model: str,
        dimensions: int,
        revision: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 64,
        timeout: float = 60.0,
        send_dimensions: bool = True,
    ) -> None:
        if not api_key or batch_size < 1 or timeout <= 0:
            raise ValueError("api_key, positive batch_size, and positive timeout are required")
        self.base_url = base_url.rstrip("/")
        self.spec = EmbeddingSpec(
            provider=self.base_url, model=model, dimensions=dimensions, revision=revision
        )
        self.api_key, self.batch_size, self.timeout = api_key, batch_size, timeout
        self.send_dimensions = send_dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        import httpx

        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                body: dict[str, object] = {"model": self.spec.model, "input": batch}
                if self.send_dimensions:
                    body["dimensions"] = self.spec.dimensions
                response = await client.post(
                    self.base_url + "/embeddings",
                    json=body,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                response.raise_for_status()
                rows = sorted(response.json()["data"], key=lambda row: row["index"])
                if [row["index"] for row in rows] != list(range(len(batch))):
                    raise ValueError("embedding response has missing or repeated indices")
                vectors.extend(row["embedding"] for row in rows)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]


class SentenceTransformerEmbedder:
    """Optional local semantic baseline with explicit query/document prompts."""

    def __init__(
        self, model: str, *, revision: str, query_prompt: str = "", document_prompt: str = ""
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, revision=revision)
        self.query_prompt, self.document_prompt = query_prompt, document_prompt
        self.spec = EmbeddingSpec(
            provider="sentence-transformers",
            model=model,
            revision=f"{revision}|query={query_prompt}|document={document_prompt}",
            dimensions=self.model.get_sentence_embedding_dimension(),
        )

    async def _encode(self, texts: Sequence[str], prompt: str) -> list[list[float]]:
        result = await asyncio.to_thread(
            self.model.encode, list(texts), prompt=prompt, normalize_embeddings=True
        )
        return result.tolist()

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._encode(texts, self.document_prompt)

    async def embed_query(self, text: str) -> list[float]:
        return (await self._encode([text], self.query_prompt))[0]
