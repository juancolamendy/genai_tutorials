"""Embedding providers abstraction layer."""

from .embeddings import (
    EmbeddingProvider,
    OllamaEmbeddings,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
)

__all__ = [
    "EmbeddingProvider",
    "OllamaEmbeddings",
    "HuggingFaceEmbeddings",
    "OpenAIEmbeddings",
]
