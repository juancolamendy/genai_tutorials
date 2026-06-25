#!/usr/bin/env python3
"""Analyze retrieval quality across all golden dataset items.

Shows which items are failing and why:
- Vector search ranking
- Keyword search ranking
- RRF fusion impact
- Reference matchability
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg
from pgvector.psycopg import register_vector
from retriever import Retriever
from retriever.queries import vector_search, keyword_search
from retriever.fusion import reciprocal_rank_fusion


def calculate_overlap(reference: str, chunk_content: str) -> float:
    """Calculate word overlap percentage."""
    ref_words = set(reference.lower().split())
    chunk_words = set(chunk_content.lower().split())
    if not ref_words:
        return 0.0
    return len(ref_words & chunk_words) / len(ref_words)


def find_reference_position(results, reference: str, threshold: float = 0.7) -> int | None:
    """Find position of reference (70%+ overlap) in results. Return None if not found."""
    for i, result in enumerate(results):
        overlap = calculate_overlap(reference, result.payload.get("content", ""))
        if overlap >= threshold:
            return i + 1  # 1-indexed
    return None


def analyze_item(item_idx: int, item: dict, retriever: Retriever, top_k: int = 60, top_n: int = 10) -> dict:
    """Analyze a single golden dataset item."""
    question = item["question"]
    reference = item["reference"]
    filters = {k: v for k, v in item.get("filters", {}).items() if v}

    # Embed query
    vec_embedding = retriever._embed(question)

    # Vector search
    vec_sql, vec_params = vector_search(vec_embedding, top_k, **filters)
    with retriever._conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(vec_sql, vec_params)
        vec_results = cur.fetchall()

    # Keyword search
    kw_sql, kw_params = keyword_search(question, top_k, **filters)
    with retriever._conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(kw_sql, kw_params)
        kw_results = cur.fetchall()

    # RRF fusion
    rrf_results = reciprocal_rank_fusion(vec_results, kw_results)[:top_n]

    # Find reference position in each ranking
    vec_pos = next((i+1 for i, r in enumerate(vec_results[:top_n])
                    if calculate_overlap(reference, r.get("content", "")) >= 0.7), None)
    kw_pos = next((i+1 for i, r in enumerate(kw_results[:top_n])
                   if calculate_overlap(reference, r.get("content", "")) >= 0.7), None)
    rrf_pos = find_reference_position(rrf_results, reference, threshold=0.7)

    # Find best overlaps
    best_vec_overlap = max((calculate_overlap(reference, r.get("content", ""))
                            for r in vec_results[:top_n]), default=0.0)
    best_kw_overlap = max((calculate_overlap(reference, r.get("content", ""))
                           for r in kw_results[:top_n]), default=0.0)
    best_rrf_overlap = max((calculate_overlap(reference, r.payload.get("content", ""))
                            for r in rrf_results), default=0.0)

    return {
        "item": item_idx,
        "question": question[:60],
        "vec_pos": vec_pos,
        "vec_overlap": best_vec_overlap,
        "kw_pos": kw_pos,
        "kw_overlap": best_kw_overlap,
        "rrf_pos": rrf_pos,
        "rrf_overlap": best_rrf_overlap,
        "found": rrf_pos is not None,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Analyze retrieval quality across golden dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=Path, default=Path("src/eval/golden_dataset.json"))
    p.add_argument("--database-url", default=os.environ.get("DATABASE_URL", ""))
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--top-k", type=int, default=60)
    args = p.parse_args(argv)

    if not args.database_url:
        print("ERROR: DATABASE_URL not set")
        return 1

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)
    print(f"Analyzing {len(dataset)} items...\n")

    # Analyze
    results = []
    with Retriever.from_credentials(args.database_url) as r:
        for i, item in enumerate(dataset):
            result = analyze_item(i, item, r, args.top_k, args.top_n)
            results.append(result)

            # Progress indicator
            status = "✓" if result["found"] else "✗"
            print(f"[{i:2d}] {status} Vec:{result['vec_pos']} Kw:{result['kw_pos']} "
                  f"RRF:{result['rrf_pos']} | {result['question']}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    found_count = sum(1 for r in results if r["found"])
    print(f"\nFound in RRF top-{args.top_n}: {found_count}/{len(results)} ({found_count/len(results)*100:.1f}%)")

    # Breakdown by position
    positions = [r["rrf_pos"] for r in results if r["found"]]
    if positions:
        print(f"\nPosition distribution (when found):")
        for pos in range(1, 6):
            count = sum(1 for p in positions if p == pos)
            pct = count / len(positions) * 100 if positions else 0
            print(f"  Position {pos}: {count} items ({pct:.0f}%)")

    # Identify problem areas
    print(f"\nProblems:")
    vec_only = sum(1 for r in results if r["vec_pos"] and not r["rrf_pos"])
    kw_only = sum(1 for r in results if r["kw_pos"] and not r["rrf_pos"])
    neither = sum(1 for r in results if not r["vec_pos"] and not r["kw_pos"])
    rrf_hurt = sum(1 for r in results if (r["vec_pos"] or r["kw_pos"]) and not r["rrf_pos"])

    print(f"  Vector found but RRF lost: {vec_only}")
    print(f"  Keyword found but RRF lost: {kw_only}")
    print(f"  Neither vector nor keyword found: {neither}")
    print(f"  RRF fusion hurt ranking: {rrf_hurt}")

    # Detailed failures
    print(f"\nFailed items (not found in RRF top-{args.top_n}):")
    for r in results:
        if not r["found"]:
            print(f"  [{r['item']:2d}] {r['question']}")
            print(f"       Vec: pos={r['vec_pos']}, overlap={r['vec_overlap']:.1%} | "
                  f"Kw: pos={r['kw_pos']}, overlap={r['kw_overlap']:.1%}")

    print("\n" + "=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
