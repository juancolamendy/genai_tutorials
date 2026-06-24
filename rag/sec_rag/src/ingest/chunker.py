#!/usr/bin/env python3
"""Chunk SEC Markdown filings using Docling's HybridChunker.

Reads Markdown files produced by converter.py (default: data/markdown),
converts each to a DoclingDocument, chunks it with HybridChunker wired to
OpenAI's tiktoken tokenizer, and writes two flat JSONL files:

  data/documents.jsonl  — one record per filing
  data/chunks.jsonl     — one record per chunk, content ready for embedding

Sidecar metadata
----------------
Place a ``<stem>.meta.json`` next to each Markdown file to supply filing-level
fields.  Any recognised key overrides what the cover-page parser finds.

    {
      "cik": "0000320193",
      "ticker": "AAPL",
      "company": "Apple Inc.",
      "form_type": "10-K",
      "fiscal_period": "FY2023",
      "filing_date": "2023-11-03"
    }

Dependencies
------------
    uv add docling "docling-core[chunking-openai]" tiktoken

Usage
-----
    uv run python chunker.py
    uv run python chunker.py --input-dir data/markdown --max-tokens 800
    uv run python chunker.py --force
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import uuid
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chunker")

# text-embedding-3-large uses cl100k_base (same as gpt-4o).
# Hard embedding limit is 8 191 tokens; 800 keeps chunks semantically tight.
EMBED_ENCODING = "cl100k_base"
DEFAULT_MAX_TOKENS = 800


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
@dataclass
class FilingMetadata:
    cik: str | None = None
    ticker: str | None = None
    company: str | None = None
    form_type: str | None = None
    fiscal_period: str | None = None
    filing_date: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def merged_with(self, other: "FilingMetadata") -> "FilingMetadata":
        base = dataclasses.asdict(self)
        for k, v in dataclasses.asdict(other).items():
            if base.get(k) is None and v is not None:
                base[k] = v
        return FilingMetadata(**base)


_FORM_RE = re.compile(
    r"\bFORM\s+(10-K(?:/A)?|10-Q(?:/A)?|8-K(?:/A)?|20-F|40-F|S-1|S-3|6-K|DEF\s?14A)\b",
    re.IGNORECASE,
)
_FY_RE = re.compile(
    r"fiscal\s+year\s+ended\s+([A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
_QTR_RE = re.compile(
    r"quarter(?:ly)?\s+period\s+ended\s+([A-Z][a-z]+\.?\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)


def _parse_cover(text: str) -> FilingMetadata:
    head = text[:6000]
    meta = FilingMetadata()
    if m := _FORM_RE.search(head):
        meta.form_type = m.group(1).upper().replace(" ", "")
    if m := _FY_RE.search(head):
        meta.fiscal_period = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(",")
    elif m := _QTR_RE.search(head):
        meta.fiscal_period = re.sub(r"\s+", " ", m.group(1)).strip().rstrip(",")
    return meta


def load_metadata(md_path: Path, cover_text: str) -> FilingMetadata:
    """Sidecar JSON is authoritative; cover-page parsing fills any gaps."""
    sidecar = md_path.with_suffix(".meta.json")
    sidecar_meta = FilingMetadata()
    if sidecar.is_file():
        try:
            raw = json.loads(sidecar.read_text(encoding="utf-8"))
            known = {f.name for f in dataclasses.fields(FilingMetadata)}
            sidecar_meta = FilingMetadata(**{k: v for k, v in raw.items() if k in known})
        except (json.JSONDecodeError, TypeError) as exc:
            log.warning("Bad sidecar %s: %s", sidecar, exc)
    return sidecar_meta.merged_with(_parse_cover(cover_text))


# --------------------------------------------------------------------------- #
# IDs
# --------------------------------------------------------------------------- #
def new_id() -> str:
    """Return a random UUID4 as a canonical hyphenated string."""
    return str(uuid.uuid4())


# --------------------------------------------------------------------------- #
# Per-file processing
# --------------------------------------------------------------------------- #
def process_file(
    md_path: Path,
    input_dir: Path,
    converter: DocumentConverter,
    chunker: HybridChunker,
) -> tuple[dict, list[dict]]:
    source_rel = md_path.relative_to(input_dir).as_posix()

    # Re-parse Markdown into a DoclingDocument so the native chunker
    # can operate on the document structure (headings, tables, lists).
    doc = converter.convert(md_path).document

    cover_text = md_path.read_text(encoding="utf-8")
    meta = load_metadata(md_path, cover_text)
    d_id = new_id()

    document_record = {
        "doc_id": d_id,
        "source_file": source_rel,
        **meta.to_dict(),
    }

    chunk_records: list[dict] = []
    for index, chunk in enumerate(chunker.chunk(dl_doc=doc)):
        # chunk.text        — raw text, no heading context
        # contextualize()   — heading-prefixed text; pass this to the embedding API
        content = chunker.contextualize(chunk=chunk)
        headings: list[str] = list(chunk.meta.headings or [])
        captions: list[str] = list(chunk.meta.captions or [])

        chunk_records.append(
            {
                "chunk_id": new_id(),
                "doc_id": d_id,
                "chunk_index": index,
                "section": " > ".join(headings) if headings else None,
                "headings": headings,
                "captions": captions,
                # This is the string sent to text-embedding-3-large.
                "content": content,
            }
        )

    return document_record, chunk_records


# --------------------------------------------------------------------------- #
# JSONL output
# --------------------------------------------------------------------------- #
def append_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chunk SEC Markdown filings with Docling HybridChunker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir", type=Path, default=Path("data/markdown"),
        help="Directory containing .md files from converter.py.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("data"),
        help="Directory to write documents.jsonl and chunks.jsonl.",
    )
    p.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help="Max tokens per chunk (embedding limit is 8191).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Delete existing output files before starting.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        log.error("Input directory not found: %s", input_dir)
        return 1

    md_files = sorted(p for p in input_dir.rglob("*.md") if p.is_file())
    if not md_files:
        log.warning("No .md files found under %s", input_dir)
        return 0

    log.info("Found %d Markdown file(s) in %s", len(md_files), input_dir)

    # Tokenizer aligned to text-embedding-3-large.
    enc = tiktoken.get_encoding(EMBED_ENCODING)
    tokenizer = OpenAITokenizer(tokenizer=enc, max_tokens=args.max_tokens)

    # HybridChunker:
    #   merge_peers=True          — merge undersized adjacent chunks sharing the same heading
    #   repeat_table_header=True  — repeat table headers in every split chunk
    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        repeat_table_header=True,
    )

    # Restrict to Markdown so Docling skips ML models (layout, OCR).
    converter = DocumentConverter(allowed_formats=[InputFormat.MD])

    docs_path = output_dir / "documents.jsonl"
    chunks_path = output_dir / "chunks.jsonl"

    if args.force:
        docs_path.unlink(missing_ok=True)
        chunks_path.unlink(missing_ok=True)
        log.info("--force: cleared existing output files.")

    total_docs = total_chunks = failed = 0
    started = time.perf_counter()

    for i, md_path in enumerate(md_files, start=1):
        log.info("[%d/%d] %s", i, len(md_files), md_path.name)
        try:
            doc_record, chunk_records = process_file(
                md_path, input_dir, converter, chunker
            )
            append_jsonl([doc_record], docs_path)
            append_jsonl(chunk_records, chunks_path)
            total_docs += 1
            total_chunks += len(chunk_records)
            log.info("  -> %d chunks", len(chunk_records))
        except Exception:  # noqa: BLE001
            log.exception("  Failed: %s", md_path)
            failed += 1

    elapsed = time.perf_counter() - started
    log.info(
        "Done in %.1fs — documents: %d, chunks: %d, failed: %d",
        elapsed, total_docs, total_chunks, failed,
    )
    log.info("Output: %s", docs_path)
    log.info("Output: %s", chunks_path)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())