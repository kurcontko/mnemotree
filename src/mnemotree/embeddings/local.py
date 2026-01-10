from __future__ import annotations

import asyncio

import numpy as np
from langchain_core.embeddings.embeddings import Embeddings


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for local embeddings. "
                "Install mnemotree with the 'lite' extra or add the package."
            ) from exc
        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._encode(texts)
        return [vector.tolist() for vector in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []

    async def aembed_query(self, text: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        encode_kwargs = {"batch_size": self.batch_size}
        try:
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=self.normalize,
                **encode_kwargs,
            )
            return np.asarray(embeddings)
        except TypeError:
            embeddings = np.asarray(self.model.encode(texts, **encode_kwargs))
            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeddings = embeddings / norms
            return embeddings
