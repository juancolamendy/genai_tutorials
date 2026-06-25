#!/usr/bin/env python3
"""Debug retrieval: manually retrieve and inspect results for a golden dataset item.

Shows:
- What chunks were retrieved (ranked by RRF score)
- Why they were ranked that way (vector score + FTS score)
- How they compare to the reference answer

Usage
-----
    uv run python debug_retrieval.py --item 0
    uv run python debug_retrieval.py --item 0 --top-n 15 --top-k 60
    uv run python debug_retrieval.py --question "What are Apple's main risks?"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg
from retriever import Retriever
from retriever.queries import vector_search, keyword_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("debug_retrieval")


def load_golden_dataset(path: Path) -> list[dict]:
    """Load golden dataset."""
    with open(path) as f:
        return json.load(f)


def calculate_overlap(reference: str, chunk_content: str) -> float:
    """Calculate % of reference words in chunk."""
    ref_words = set(reference.lower().split())
    chunk_words = set(chunk_content.lower().split())

    if not ref_words:
        return 0.0

    overlap = len(ref_words & chunk_words)
    return overlap / len(ref_words)


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def print_item_details(item: dict, item_index: int | None = None) -> None:
    """Print question and reference from a dataset item."""
    if item_index is not None:
        print(f"\n[Item {item_index}] Golden Dataset Entry")
    else:
        print(f"\n[Custom Question]")
    print("─" * 80)
    print(f"Question:  {item['question']}")
    print(f"\nReference: {item['reference']}")
    if "filters" in item:
        print(f"\nFilters:   {item['filters']}")


def print_search_results(
    label: str,
    results: list[dict],
    reference: str,
    max_show: int = 5,
) -> None:
    """Print results from a single search method (vector or keyword)."""
    if not results:
        print(f"\n{label}: 0 results ✗")
        return

    print(f"\n{label}: {len(results)} results found\n")

    for i, result in enumerate(results[:max_show], 1):
        content = result.get("content", "")
        overlap = calculate_overlap(reference, content)
        rank_score = result.get("cos_dist", result.get("fts_rank", "N/A"))

        # Format score label
        score_label = "cos_dist" if "cos_dist" in result else "fts_rank"
        status = "✓" if overlap >= 0.7 else "~" if overlap >= 0.4 else "✗"

        print(f"[{i}] {status} Overlap: {overlap:.1%} | {score_label}: {rank_score}")
        print(f"    Content: {content[:80]}...")
        print()

    if len(results) > max_show:
        print(f"    ... and {len(results) - max_show} more")


def print_retrieved_chunks(
    results: list,
    reference: str,
) -> None:
    """Print final RRF-fused results."""
    if not results:
        print("\n✗ No chunks retrieved!")
        return

    print(f"\nFinal RRF-Fused Results ({len(results)} unique chunks):\n")

    for i, result in enumerate(results, 1):
        payload = result.payload
        content = payload.get("content", "")
        overlap = calculate_overlap(reference, content)

        status = "✓" if overlap >= 0.7 else "~" if overlap >= 0.4 else "✗"

        print(f"[{i}] {status} Overlap: {overlap:.1%} | RRF Score: {result.rrf_score:.4f}")
        print(f"    Content: {content[:100]}...")
        print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Debug retrieval on a specific golden dataset item.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=Path, default=Path("src/eval/golden_dataset.json"),
                   help="Path to golden dataset JSON.")
    p.add_argument("--item", type=int, default=None,
                   help="Index of item in golden dataset to retrieve (0-based).")
    p.add_argument("--question", type=str, default=None,
                   help="Custom question to retrieve (if not using --item).")
    p.add_argument("--ticker", type=str, default=None,
                   help="Optional ticker filter.")
    p.add_argument("--form-type", type=str, default=None,
                   help="Optional form type filter.")
    p.add_argument("--fiscal-period", type=str, default=None,
                   help="Optional fiscal period filter.")
    p.add_argument("--top-n", type=int, default=15,
                   help="Number of chunks to retrieve.")
    p.add_argument("--top-k", type=int, default=60,
                   help="Candidate pool size per retriever.")
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL", ""),
                   help="Database URL (or DATABASE_URL env var).")
    p.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                   help="OpenAI API key (or OPENAI_API_KEY env var).")
    args = p.parse_args(argv)

    if not args.openai_api_key:
        log.error("OPENAI_API_KEY is not set.")
        return 1
    if not args.database_url:
        log.error("DATABASE_URL is not set.")
        return 1

    # Load item or question
    if args.item is not None:
        if not args.dataset.exists():
            log.error("Golden dataset not found: %s", args.dataset)
            return 1
        dataset = load_golden_dataset(args.dataset)
        if args.item < 0 or args.item >= len(dataset):
            log.error("Item index out of range: %d (dataset has %d items)", args.item, len(dataset))
            return 1
        item = dataset[args.item]
    elif args.question:
        item = {
            "question": args.question,
            "reference": "(custom question — no reference provided)",
            "filters": {
                "ticker": args.ticker,
                "form_type": args.form_type,
                "fiscal_period": args.fiscal_period,
            },
        }
    else:
        log.error("Either --item or --question is required.")
        p.print_help()
        return 1

    print_header("RETRIEVER DEBUG SESSION")
    print_item_details(item, args.item if args.item is not None else None)

    # Retrieve
    log.info("Retrieving chunks with top_k=%d, top_n=%d...", args.top_k, args.top_n)

    filters = item.get("filters", {})
    filters = {k: v for k, v in filters.items() if v}  # Remove None values

    with Retriever.from_credentials(args.database_url, args.openai_api_key) as r:
        # Embed once
        log.info("Embedding query…")
        vec_embedding = r._embed(item["question"])

        # Run vector search
        log.info("Running vector search…")
        vec_sql, vec_params = vector_search(vec_embedding, args.top_k, **filters)
        with r._conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(vec_sql, vec_params)
            vec_results = cur.fetchall()

        # Run keyword search
        log.info("Running keyword search…")
        kw_sql, kw_params = keyword_search(item["question"], args.top_k, **filters)
        with r._conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(kw_sql, kw_params)
            kw_results = cur.fetchall()

        # Fuse manually (don't call retrieve to avoid re-embedding)
        log.info("Fusing with RRF…")
        from retriever.fusion import reciprocal_rank_fusion
        results = reciprocal_rank_fusion(vec_results, kw_results)[:args.top_n]

    print_search_results("Vector Search (top 5)", vec_results[:5], item["reference"])
    print_search_results("Keyword Search (top 5)", kw_results[:5], item["reference"])
    print_retrieved_chunks(results, item["reference"])

    # Summary
    reference = item["reference"]
    overlaps = [calculate_overlap(reference, r.payload["content"]) for r in results]
    max_overlap = max(overlaps) if overlaps else 0.0
    matched_chunks = sum(1 for o in overlaps if o >= 0.7)

    print("─" * 80)
    print(f"Summary:")
    print(f"  Chunks with 70%+ overlap: {matched_chunks}/{len(results)}")
    print(f"  Best match: {max_overlap:.1%}")
    if matched_chunks == 0:
        print(f"  → Reference NOT FOUND in top-{args.top_n} results")
    else:
        print(f"  → Reference found at position {next(i+1 for i, o in enumerate(overlaps) if o >= 0.7)}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
