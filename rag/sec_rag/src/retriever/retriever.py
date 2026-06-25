"""Hybrid retriever for the SEC RAG store.

Combines semantic search (pgvector halfvec cosine) and keyword search
(Postgres full-text) via Reciprocal Rank Fusion (RRF).

Pipeline
--------
1. Embed the user query with text-embedding-3-large (same model used at
   index time — the vectors must live in the same space).
2. Run vector_search  → top-K chunks by cosine similarity.
3. Run keyword_search → top-K chunks by ts_rank (BM25-like full-text).
4. Fuse both ranked lists with RRF → single ranked list.
5. Return the top-N fused results with their metadata.

Modules consumed
----------------
queries.py  — SQL builders (vector_search, keyword_search)
fusion.py   — reciprocal_rank_fusion

Environment variables
---------------------
OPENAI_API_KEY   required
DATABASE_URL     required  e.g. postgresql://user:pass@localhost:5432/mydb

Usage (library)
---------------
    from retriever import Retriever

    r = Retriever.from_env()
    results = r.retrieve("What are Apple's main supply chain risks?",
                         ticker="AAPL", form_type="10-K", top_n=10)
    for res in results:
        print(res.rrf_score, res.payload["section"])
        print(res.payload["content"])

Usage (CLI)
-----------
    uv run python retriever.py "What are Apple's supply chain risks?" \\
        --ticker AAPL --form-type 10-K --top-n 10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
import time

import psycopg
from pgvector.psycopg import register_vector

try:
    # Running as module
    from .fusion import RankedResult, reciprocal_rank_fusion
    from .queries import keyword_search, vector_search
    from embeddings import EmbeddingProvider
except ImportError:
    # Running as script directly
    from fusion import RankedResult, reciprocal_rank_fusion  # type: ignore
    from queries import keyword_search, vector_search  # type: ignore
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from embeddings import EmbeddingProvider  # type: ignore

log = logging.getLogger("retriever")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

# How many candidates each ranker fetches before fusion.
# Wider pools improve recall at the cost of two slightly larger DB round-trips.
RETRIEVER_TOP_K = 40


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class Retriever:
    """Hybrid retriever: vector search + full-text search fused with RRF."""

    def __init__(self, conn: psycopg.Connection, embedding_provider: EmbeddingProvider) -> None:
        self._conn = conn
        self._embedding_provider = embedding_provider
        self._embedding_cache: dict[str, list[float]] = {}  # query → embedding

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Retriever":
        """Construct from DATABASE_URL environment variable.

        Embedding provider is determined by EmbeddingProvider.from_env(),
        which checks EMBEDDING_PROVIDER and related env vars.
        """
        database_url = os.environ.get("DATABASE_URL", "")
        if not database_url:
            raise RuntimeError("DATABASE_URL is not set.")
        return cls.from_credentials(database_url)

    @classmethod
    def from_credentials(cls, database_url: str) -> "Retriever":
        """Create retriever from database URL.

        Embedding provider is auto-detected via EmbeddingProvider.from_env().
        """
        conn = psycopg.connect(database_url)
        register_vector(conn)   # register halfvec ↔ Python adapter
        embedding_provider = EmbeddingProvider.from_env()
        log.info(f"Using embedding provider: {embedding_provider.__class__.__name__}")
        return cls(conn, embedding_provider)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def _embed(self, text: str) -> list[float]:
        """Embed a single query string.

        Uses the configured embedding provider (Ollama, HuggingFace, or OpenAI).
        Results are cached in memory to avoid re-embedding identical queries.
        """
        # Check cache first
        if text in self._embedding_cache:
            log.debug("Embedding cache hit: %r", text[:50])
            return self._embedding_cache[text]

        # Embed using the provider
        log.debug("Embedding query: %r", text[:50])
        embedding = self._embedding_provider.embed(text)
        self._embedding_cache[text] = embedding
        return embedding

    # ------------------------------------------------------------------
    # Individual retrievers
    # ------------------------------------------------------------------
    def _run_vector_search(
        self,
        query_embedding: list[float],
        top_k: int,
        ticker: str | None,
        form_type: str | None,
        fiscal_period: str | None,
    ) -> list[dict]:
        sql, params = vector_search(
            query_embedding=query_embedding,
            top_k=top_k,
            ticker=ticker,
            form_type=form_type,
            fiscal_period=fiscal_period,
        )
        with self._conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def _run_keyword_search(
        self,
        query_text: str,
        top_k: int,
        ticker: str | None,
        form_type: str | None,
        fiscal_period: str | None,
    ) -> list[dict]:
        sql, params = keyword_search(
            query_text=query_text,
            top_k=top_k,
            ticker=ticker,
            form_type=form_type,
            fiscal_period=fiscal_period,
        )
        with self._conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        ticker: str | None = None,
        form_type: str | None = None,
        fiscal_period: str | None = None,
        top_k: int = RETRIEVER_TOP_K,
        top_n: int = 10,
    ) -> list[RankedResult]:
        """Run hybrid retrieval and return top-N fused results.

        Parameters
        ----------
        query:
            Natural-language user question.
        ticker:
            Optional filter — e.g. ``"AAPL"``.
        form_type:
            Optional filter — e.g. ``"10-K"``.
        fiscal_period:
            Optional filter — e.g. ``"FY2023"``.
        top_k:
            Candidate pool size for each retriever before fusion.
        top_n:
            Final number of results to return after fusion.

        Raises
        ------
        ValueError
            If query is empty or whitespace-only.
        """
        # Validate query
        query_clean = query.strip()
        if not query_clean:
            raise ValueError("Query cannot be empty or whitespace-only.")

        log.info("Embedding query…")
        embedding = self._embed(query_clean)

        log.info("Running vector search (top_k=%d)…", top_k)
        vec_results = self._run_vector_search(
            embedding, top_k, ticker, form_type, fiscal_period
        )
        log.info("  → %d vector hits", len(vec_results))

        log.info("Running keyword search (top_k=%d)…", top_k)
        kw_results = self._run_keyword_search(
            query_clean, top_k, ticker, form_type, fiscal_period
        )
        log.info("  → %d keyword hits", len(kw_results))

        log.info("Fusing with RRF…")
        fused = reciprocal_rank_fusion(vec_results, kw_results)
        log.info("  → %d unique chunks after fusion, returning top %d", len(fused), top_n)

        return fused[:top_n]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_results(results: list[RankedResult]) -> None:
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, start=1):
        p = r.payload
        print(f"\n{'─' * 72}")
        print(
            f"[{i}] score={r.rrf_score:.5f}  "
            f"{p.get('ticker', '?')} {p.get('form_type', '?')} "
            f"{p.get('fiscal_period', '?')}"
        )
        print(
            f"     doc_id : {p.get('document_id', '?')}  "
            f"chunk_id: {r.chunk_id}"
        )
        print(f"     chunk  : {p.get('chunk_index', '?')}  "
              f"url: {p.get('url', '?')[:60]}...")
        print()
        preview = textwrap.fill(
            p.get("content", "")[:400],
            width=72,
            initial_indent="  ",
            subsequent_indent="  ",
        )
        print(preview)
    print(f"\n{'─' * 72}")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid retriever: vector + full-text search with RRF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("query", help="Natural-language question to retrieve for.")
    p.add_argument("--ticker",        default=None, help="Filter by ticker, e.g. AAPL.")
    p.add_argument("--form-type",     default=None, help="Filter by form type, e.g. 10-K.")
    p.add_argument("--fiscal-period", default=None, help="Filter by fiscal period, e.g. FY2023.")
    p.add_argument("--top-k",  type=int, default=RETRIEVER_TOP_K,
                   help="Candidate pool per retriever before fusion.")
    p.add_argument("--top-n",  type=int, default=10,
                   help="Final results to display after fusion.")
    p.add_argument("--database-url",   default=os.environ.get("DATABASE_URL", ""),
                   help="Postgres DSN (default: $DATABASE_URL).")
    p.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                   help="OpenAI key (default: $OPENAI_API_KEY).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.database_url:
        log.error("DATABASE_URL not set. Use --database-url or export the env var.")
        return 1
    if not args.openai_api_key:
        log.error("OPENAI_API_KEY not set. Use --openai-api-key or export the env var.")
        return 1

    with Retriever.from_credentials(args.database_url, args.openai_api_key) as r:
        results = r.retrieve(
            args.query,
            ticker=args.ticker,
            form_type=args.form_type,
            fiscal_period=args.fiscal_period,
            top_k=args.top_k,
            top_n=args.top_n,
        )

    _print_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())