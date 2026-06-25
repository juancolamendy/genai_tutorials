#!/usr/bin/env python3
"""Migrate from OpenAI embeddings (3072-dim) to Nomic embeddings (768-dim).

This script:
1. Drops the old chunks table with 3072-dim vectors
2. Recreates it with 768-dim vectors (nomic-embed-text)
3. Re-ingests and re-embeds all chunks from the source JSONL files

Usage:
    uv run python scripts/migrate_to_nomic_embeddings.py
    uv run python scripts/migrate_to_nomic_embeddings.py --chunks-dir data/chunks
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid as uuid_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import psycopg
from pgvector.psycopg import register_vector

from embeddings import EmbeddingProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("migrate_embeddings")


def load_chunks(path: Path) -> list[dict]:
    """Load chunks from JSONL."""
    chunks = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            chunks.append(json.loads(line))
    log.info(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def load_documents(path: Path) -> dict[str, dict]:
    """Load documents from JSONL and index by ID."""
    docs = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            docs[data["id"]] = data
    log.info(f"Loaded {len(docs)} documents from {path}")
    return docs


def drop_and_recreate_schema(conn: psycopg.Connection, dim: int = 768) -> None:
    """Drop old chunks/documents tables and recreate with new vector dimension."""
    with conn.cursor() as cur:
        log.info("Dropping old schema...")
        cur.execute("DROP TABLE IF EXISTS chunks CASCADE;")
        cur.execute("DROP TABLE IF EXISTS documents CASCADE;")
        cur.execute("DROP EXTENSION IF EXISTS vector CASCADE;")

        log.info("Creating pgvector extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        log.info(f"Creating documents table...")
        cur.execute("""
            CREATE TABLE documents (
                id UUID PRIMARY KEY,
                ticker TEXT,
                company TEXT,
                form_type TEXT NOT NULL,
                fiscal_period TEXT,
                filing_date DATE,
                url TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        log.info(f"Creating chunks table with {dim}-dim vectors...")
        cur.execute(f"""
            CREATE TABLE chunks (
                id UUID PRIMARY KEY,
                document_id UUID NOT NULL REFERENCES documents(id),
                section TEXT,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector({dim}) NOT NULL,
                tsv tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', COALESCE(content, ''))
                ) STORED,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        log.info("Creating indexes...")
        cur.execute("""
            CREATE INDEX idx_chunks_document_id ON chunks(document_id);
        """)
        cur.execute("""
            CREATE INDEX idx_chunks_section ON chunks(section);
        """)
        cur.execute("""
            CREATE INDEX chunks_tsv_idx ON chunks USING GIN(tsv);
        """)

        # HNSW index for vector search (build after data is loaded)
        log.info(f"Creating HNSW index for {dim}-dim vectors...")
        cur.execute(f"""
            CREATE INDEX chunks_embedding_hnsw ON chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)

        conn.commit()
        log.info("Schema created successfully")


def ingest_documents(
    conn: psycopg.Connection,
    documents: dict[str, dict],
) -> None:
    """Insert documents into database."""
    log.info(f"Ingesting {len(documents)} documents...")

    with conn.cursor() as cur:
        for doc_id, doc in documents.items():
            cur.execute("""
                INSERT INTO documents (id, ticker, company, form_type, fiscal_period, filing_date, url)
                VALUES (%(id)s, %(ticker)s, %(company)s, %(form_type)s, %(fiscal_period)s, %(filing_date)s, %(url)s)
                ON CONFLICT(id) DO NOTHING;
            """, {
                "id": uuid_module.UUID(doc_id),
                "ticker": doc.get("ticker"),
                "company": doc.get("company"),
                "form_type": doc.get("form_type"),
                "fiscal_period": doc.get("fiscal_period"),
                "filing_date": doc.get("filing_date"),
                "url": doc.get("url"),
            })

    conn.commit()
    log.info("Documents ingested")


def ingest_chunks(
    conn: psycopg.Connection,
    chunks: list[dict],
    embedding_provider: EmbeddingProvider,
) -> None:
    """Embed chunks and insert into database."""
    log.info(f"Embedding and ingesting {len(chunks)} chunks...")

    register_vector(conn)

    with conn.cursor() as cur:
        for i, chunk in enumerate(chunks, start=1):
            if i % 100 == 0:
                log.info(f"  [{i}/{len(chunks)}] Processing chunk {chunk['id']}")

            # Embed the chunk content
            content = chunk.get("content", "")
            if not content:
                log.warning(f"Skipping chunk {chunk['id']}: empty content")
                continue

            embedding = embedding_provider.embed(content)

            # Insert chunk
            cur.execute("""
                INSERT INTO chunks (id, document_id, section, chunk_index, content, embedding)
                VALUES (%(id)s, %(document_id)s, %(section)s, %(chunk_index)s, %(content)s, %(embedding)s)
                ON CONFLICT(id) DO NOTHING;
            """, {
                "id": uuid_module.UUID(chunk["id"]),
                "document_id": uuid_module.UUID(chunk["document_id"]),
                "section": chunk.get("section"),
                "chunk_index": chunk.get("chunk_index"),
                "content": content,
                "embedding": embedding,
            })

            # Commit in batches
            if i % 100 == 0:
                conn.commit()

    conn.commit()
    log.info(f"All {len(chunks)} chunks ingested and embedded")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Migrate from OpenAI embeddings (3072-dim) to Nomic embeddings (768-dim)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--chunks-dir", type=Path, default=Path("data/chunks"),
        help="Directory containing chunks.jsonl and documents.jsonl",
    )
    p.add_argument(
        "--embedding-provider", default=os.environ.get("EMBEDDING_PROVIDER", "ollama"),
        help="Embedding provider (ollama, huggingface, openai)",
    )
    p.add_argument(
        "--no-confirm", action="store_true",
        help="Skip confirmation prompt (DANGEROUS: will delete existing data)",
    )
    args = p.parse_args(argv)

    # Validate inputs
    chunks_path = args.chunks_dir / "chunks.jsonl"
    docs_path = args.chunks_dir / "documents.jsonl"

    if not chunks_path.exists():
        log.error(f"Chunks file not found: {chunks_path}")
        return 1
    if not docs_path.exists():
        log.error(f"Documents file not found: {docs_path}")
        return 1

    # Get database URL
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        log.error("DATABASE_URL not set")
        return 1

    # Confirm dangerous operation
    if not args.no_confirm:
        print("\n⚠️  This will:")
        print("   1. DROP the existing chunks and documents tables")
        print("   2. Delete all existing embeddings")
        print("   3. Re-embed all chunks with the new embedding provider")
        print()
        response = input("Continue? (type 'yes' to proceed): ").strip().lower()
        if response != "yes":
            log.info("Cancelled")
            return 0

    # Load data
    log.info("Loading source data...")
    chunks = load_chunks(chunks_path)
    documents = load_documents(docs_path)

    # Get embedding provider
    log.info(f"Using embedding provider: {args.embedding_provider}")
    try:
        if args.embedding_provider == "ollama":
            from embeddings import OllamaEmbeddings
            embedding_provider = OllamaEmbeddings()
        elif args.embedding_provider == "huggingface":
            from embeddings import HuggingFaceEmbeddings
            embedding_provider = HuggingFaceEmbeddings()
        elif args.embedding_provider == "openai":
            from embeddings import OpenAIEmbeddings
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                log.error("OPENAI_API_KEY not set")
                return 1
            embedding_provider = OpenAIEmbeddings(api_key=api_key)
        else:
            log.error(f"Unknown provider: {args.embedding_provider}")
            return 1
    except Exception as e:
        log.error(f"Failed to create embedding provider: {e}")
        return 1

    # Get vector dimension
    vec_dim = embedding_provider.dimension
    log.info(f"Embedding dimension: {vec_dim}")

    # Connect to database
    log.info("Connecting to database...")
    conn = psycopg.connect(database_url)

    try:
        # Migrate schema
        log.info("Migrating database schema...")
        drop_and_recreate_schema(conn, dim=vec_dim)

        # Ingest data
        ingest_documents(conn, documents)
        ingest_chunks(conn, chunks, embedding_provider)

        log.info("✓ Migration complete!")
        print("\n" + "=" * 70)
        print("MIGRATION COMPLETE")
        print("=" * 70)
        print(f"Documents: {len(documents)}")
        print(f"Chunks: {len(chunks)}")
        print(f"Embedding dimension: {vec_dim}")
        print(f"Embedding provider: {embedding_provider.__class__.__name__}")
        print("=" * 70 + "\n")

        return 0

    except Exception as e:
        log.error(f"Migration failed: {e}", exc_info=True)
        return 1

    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
