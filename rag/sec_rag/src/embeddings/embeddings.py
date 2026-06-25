"""Embedding providers abstraction layer.

Supports multiple embedding backends:
- Ollama (local, GPU-accelerated)
- HuggingFace (local, pure Python)
- OpenAI (cloud, legacy)

Usage:
    provider = EmbeddingProvider.from_env()
    embedding = provider.embed("query text")
"""

from __future__ import annotations

import os
import logging
import time
from abc import ABC, abstractmethod

log = logging.getLogger("embeddings")


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a vector (list of floats)."""
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Returns list of vectors."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension (e.g., 768 for nomic-embed-text)."""
        pass

    @classmethod
    def from_env(cls) -> EmbeddingProvider:
        """Create provider from environment variables.

        Priority:
        1. EMBEDDING_PROVIDER env var (ollama, huggingface, openai)
        2. Check for provider-specific config (OLLAMA_EMBED_URL, etc.)
        3. Fall back to OpenAI if OPENAI_API_KEY is set
        4. Error if no provider configured
        """
        provider_name = os.environ.get("EMBEDDING_PROVIDER", "").lower()

        if provider_name == "ollama":
            url = os.environ.get("OLLAMA_EMBED_URL", "http://localhost:11434")
            return OllamaEmbeddings(url=url)

        if provider_name == "huggingface":
            model = os.environ.get(
                "HUGGINGFACE_EMBED_MODEL",
                "nomic-ai/nomic-embed-text-1.5"
            )
            return HuggingFaceEmbeddings(model_name=model)

        if provider_name == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set but provider=openai")
            return OpenAIEmbeddings(api_key=api_key)

        # Default: Ollama (not OpenAI, even if OPENAI_API_KEY is set)
        log.info("No EMBEDDING_PROVIDER set, defaulting to Ollama")
        return OllamaEmbeddings(url=os.environ.get("OLLAMA_EMBED_URL", "http://localhost:11434"))


# --------------------------------------------------------------------------- #
# Ollama
# --------------------------------------------------------------------------- #
class OllamaEmbeddings(EmbeddingProvider):
    """Embedding provider using Ollama (local, GPU-accelerated).

    Requires: ollama pull nomic-embed-text && ollama serve
    """

    MODEL = "nomic-embed-text"
    DIMENSION = 768

    def __init__(self, url: str = "http://localhost:11434"):
        """Initialize Ollama client.

        Args:
            url: Ollama API endpoint (default: http://localhost:11434)
        """
        self.url = url.rstrip("/")
        self._test_connection()

    def _test_connection(self) -> None:
        """Verify Ollama is running and model is available."""
        import requests

        try:
            # Check if Ollama is running
            resp = requests.get(f"{self.url}/api/tags", timeout=5)
            resp.raise_for_status()
            tags = resp.json()

            # Check if model is available
            models = [m["name"] for m in tags.get("models", [])]
            if self.MODEL not in models and f"{self.MODEL}:latest" not in models:
                log.warning(
                    f"Model {self.MODEL} not found. "
                    f"Run: ollama pull {self.MODEL}"
                )
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.url}. "
                f"Make sure Ollama is running: ollama serve"
            ) from e

    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        import requests

        try:
            resp = requests.post(
                f"{self.url}/api/embed",
                json={"model": self.MODEL, "input": texts},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["embeddings"]
        except Exception as e:
            log.error(f"Ollama embedding failed: {e}")
            raise

    @property
    def dimension(self) -> int:
        return self.DIMENSION


# --------------------------------------------------------------------------- #
# HuggingFace
# --------------------------------------------------------------------------- #
class HuggingFaceEmbeddings(EmbeddingProvider):
    """Embedding provider using HuggingFace sentence-transformers.

    Requires: pip install sentence-transformers torch
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-1.5"):
        """Initialize HuggingFace embeddings.

        Args:
            model_name: Model name on HuggingFace Hub
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers torch"
            )

        log.info(f"Loading HuggingFace model: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
        )
        self.model_name = model_name

    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


# --------------------------------------------------------------------------- #
# OpenAI (Legacy)
# --------------------------------------------------------------------------- #
class OpenAIEmbeddings(EmbeddingProvider):
    """Embedding provider using OpenAI API (legacy).

    Requires: OPENAI_API_KEY environment variable
    """

    MODEL = "text-embedding-3-large"
    DIMENSION = 3072

    def __init__(self, api_key: str):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai not installed. Install with: pip install openai"
            )

        self.client = OpenAI(api_key=api_key)
        self.max_retries = 3
        self.retry_backoff_base = 2

    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts with retry logic."""
        for attempt in range(self.max_retries):
            try:
                resp = self.client.embeddings.create(
                    model=self.MODEL,
                    input=texts,
                    encoding_format="float",
                )
                return [item.embedding for item in resp.data]
            except Exception as e:
                if attempt < self.max_retries - 1:
                    backoff = self.retry_backoff_base ** attempt
                    log.warning(
                        f"Embedding error (attempt {attempt + 1}/{self.max_retries}, "
                        f"will retry in {backoff}s): {e}"
                    )
                    time.sleep(backoff)
                else:
                    log.error(f"Embedding failed after {self.max_retries} attempts")
                    raise

    @property
    def dimension(self) -> int:
        return self.DIMENSION
