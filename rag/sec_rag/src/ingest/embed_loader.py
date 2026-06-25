#!/usr/bin/env python3
"""Embed chunk content and load documents + chunks into Postgres.

Reads:
  data/documents.jsonl  — produced by chunker.py
  data/chunks.jsonl     — produced by chunker.py

For each chunk, calls the OpenAI embeddings API (text-embedding-3-large)
and upserts into the two Postgres tables defined in schema.sql:

  documents (id, cik, ticker, company, form_type, filing_date, fiscal_period, url)
  chunks    (id, document_id, section, chunk_index, content, embedding, tsv)

The ``tsv`` column is GENERATED ALWAYS in schema.sql, so we never write to it
— Postgres recomputes it automatically on every upsert.

Embedding calls are batched (default: 128 chunks per API call) to stay well
inside the 300 k-token-per-minute rate limit for text-embedding-3-large.
The script is idempotent: both upserts use ON CONFLICT DO UPDATE, so re-running
after adding new filings is safe.

Environment variables
---------------------
OPENAI_API_KEY   — required
DATABASE_URL     — required  e.g. postgresql://user:pass@localhost:5432/mydb

Usage
-----
    uv add openai psycopg[binary] pgvector
    uv run python embed_loader.py
    uv run python embed_loader.py --batch-size 64 --documents data/documents.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import uuid

import psycopg
from openai import OpenAI
from pgvector.psycopg import register_vector
from pgvector import HalfVector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("embed_loader")

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM   = 3072


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines if l.strip()]


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def embed_batch(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Call the OpenAI embeddings endpoint for a list of texts.

    Returns a list of float vectors in the same order as the input.
    Retries once on transient errors with a short back-off.
    """
    for attempt in range(2):
        try:
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=texts,
                encoding_format="float",
            )
            # The API guarantees the same order as the input.
            return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        except Exception as exc:  # noqa: BLE001
            if attempt == 0:
                log.warning("Embedding error (will retry in 5 s): %s", exc)
                time.sleep(5)
            else:
                raise


# --------------------------------------------------------------------------- #
# DB upserts
# --------------------------------------------------------------------------- #
_UPSERT_DOCUMENT = """
INSERT INTO documents (id, cik, ticker, company, form_type, filing_date, fiscal_period, url)
VALUES (%(id)s, %(cik)s, %(ticker)s, %(company)s,
        %(form_type)s, %(filing_date)s, %(fiscal_period)s, %(url)s)
ON CONFLICT (id) DO UPDATE SET
    cik           = EXCLUDED.cik,
    ticker        = EXCLUDED.ticker,
    company       = EXCLUDED.company,
    form_type     = EXCLUDED.form_type,
    filing_date   = EXCLUDED.filing_date,
    fiscal_period = EXCLUDED.fiscal_period,
    url           = EXCLUDED.url;
"""

_UPSERT_CHUNK = """
INSERT INTO chunks (id, document_id, section, chunk_index, content, embedding)
VALUES (%(id)s, %(document_id)s, %(section)s, %(chunk_index)s,
        %(content)s, %(embedding)s)
ON CONFLICT (id) DO UPDATE SET
    document_id = EXCLUDED.document_id,
    section     = EXCLUDED.section,
    chunk_index = EXCLUDED.chunk_index,
    content     = EXCLUDED.content,
    embedding   = EXCLUDED.embedding;
"""
# NOTE: tsv is a GENERATED ALWAYS column — Postgres recomputes it automatically.


def upsert_documents(conn: psycopg.Connection, docs: list[dict]) -> None:
    """Upsert all document records."""
    rows = [
        {
            "id":            uuid.UUID(d["id"]),
            "cik":           d.get("cik"),
            "ticker":        d.get("ticker"),
            "company":       d.get("company"),
            "form_type":     d.get("form_type"),
            "filing_date":   d.get("filing_date"),
            "fiscal_period": d.get("fiscal_period"),
            "url":           d["url"],  # guaranteed by converter.py metadata
        }
        for d in docs
    ]
    with conn.cursor() as cur:
        cur.executemany(_UPSERT_DOCUMENT, rows)
    conn.commit()
    log.info("Upserted %d document(s).", len(rows))


def upsert_chunks_batch(
    conn: psycopg.Connection,
    chunks: list[dict],
    vectors: list[list[float]],
) -> None:
    """Upsert a batch of chunks together with their embeddings."""
    rows = [
        {
            "id":            uuid.UUID(c["id"]),
            "document_id":   uuid.UUID(c["document_id"]),
            "section":       c.get("section"),
            "chunk_index":   c["chunk_index"],
            "content":       c["content"],
            # HalfVector converts the list[float] to the halfvec wire format
            # that psycopg + pgvector sends as a native halfvec(3072) value.
            "embedding":     HalfVector(v),
        }
        for c, v in zip(chunks, vectors)
    ]
    with conn.cursor() as cur:
        cur.executemany(_UPSERT_CHUNK, rows)
    conn.commit()


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def run(
    documents_path: Path,
    chunks_path: Path,
    batch_size: int,
    database_url: str,
    openai_api_key: str,
) -> int:
    # --- load JSONL ----------------------------------------------------------
    if not documents_path.is_file():
        log.error("Not found: %s", documents_path)
        return 1
    if not chunks_path.is_file():
        log.error("Not found: %s", chunks_path)
        return 1

    docs   = read_jsonl(documents_path)
    chunks = read_jsonl(chunks_path)
    log.info("Loaded %d documents, %d chunks.", len(docs), len(chunks))

    # --- connect + register halfvec type -------------------------------------
    conn = psycopg.connect(database_url)
    register_vector(conn)   # teaches psycopg how to send/receive vector types
    log.info("Connected to Postgres.")

    # --- upsert documents (no embedding needed) ------------------------------
    upsert_documents(conn, docs)

    # --- embed + upsert chunks in batches ------------------------------------
    client = OpenAI(api_key=openai_api_key)
    total  = len(chunks)
    done   = 0
    started = time.perf_counter()

    for start in range(0, total, batch_size):
        batch     = chunks[start : start + batch_size]
        texts     = [c["content"] for c in batch]
        batch_num = start // batch_size + 1
        n_batches = (total + batch_size - 1) // batch_size

        log.info(
            "Embedding batch %d/%d (%d chunks)…",
            batch_num, n_batches, len(texts),
        )
        vectors = embed_batch(client, texts)
        upsert_chunks_batch(conn, batch, vectors)
        done += len(batch)
        log.info("  → %d/%d chunks loaded.", done, total)

    conn.close()
    elapsed = time.perf_counter() - started
    log.info("Done in %.1fs — %d documents, %d chunks.", elapsed, len(docs), total)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed chunk.content and load documents + chunks into Postgres.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--documents", type=Path, default=Path("data/chunks/documents.jsonl"),
        help="Path to documents.jsonl from chunker.py.",
    )
    p.add_argument(
        "--chunks", type=Path, default=Path("data/chunks/chunks.jsonl"),
        help="Path to chunks.jsonl from chunker.py.",
    )
    p.add_argument(
        "--batch-size", type=int, default=128,
        help="Chunks per OpenAI embeddings API call.",
    )
    p.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", ""),
        help="Postgres connection string (default: $DATABASE_URL).",
    )
    p.add_argument(
        "--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (default: $OPENAI_API_KEY).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.database_url:
        log.error("DATABASE_URL is not set. Use --database-url or set the env var.")
        return 1
    if not args.openai_api_key:
        log.error("OPENAI_API_KEY is not set. Use --openai-api-key or set the env var.")
        return 1

    return run(
        documents_path  = args.documents,
        chunks_path     = args.chunks,
        batch_size      = args.batch_size,
        database_url    = args.database_url,
        openai_api_key  = args.openai_api_key,
    )


if __name__ == "__main__":
    sys.exit(main())