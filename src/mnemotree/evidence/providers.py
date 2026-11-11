"""The 1.0 path of :mod:`mnemotree.providers.embeddings`, kept importable until 2.0."""

from mnemotree.core.models import EmbeddingSpec as EmbeddingSpec
from mnemotree.providers.embeddings import DEFAULT_LOCAL_EMBEDDING as DEFAULT_LOCAL_EMBEDDING
from mnemotree.providers.embeddings import TOKENIZER_FILES as TOKENIZER_FILES
from mnemotree.providers.embeddings import EmbedderUnavailable as EmbedderUnavailable
from mnemotree.providers.embeddings import FastEmbedEmbedder as FastEmbedEmbedder
from mnemotree.providers.embeddings import OpenAIEmbedder as OpenAIEmbedder
from mnemotree.providers.embeddings import (
    SentenceTransformerEmbedder as SentenceTransformerEmbedder,
)
from mnemotree.providers.embeddings import logger as logger
