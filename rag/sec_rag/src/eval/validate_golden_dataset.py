#!/usr/bin/env python3
"""Validate that golden dataset references actually exist in chunks.

Checks:
1. Does the reference text appear in chunks matching the filters?
2. Is the reference substantive or just a question rephrasing?
3. What's the quality of the question-reference pair?

Usage
-----
    uv run python validate_golden_dataset.py
    uv run python validate_golden_dataset.py --dataset golden_dataset.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate_golden_dataset")


@dataclass
class ValidationResult:
    """Result of validating one dataset item."""
    item_index: int
    question: str
    reference: str
    filters: dict
    found_in_chunks: bool
    matching_chunks_count: int
    word_overlap: float
    is_substantive: bool
    issues: list[str]


def load_chunks(path: Path) -> list[dict]:
    """Load chunks from JSONL."""
    chunks = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            chunks.append(json.loads(line))
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
    return docs


def load_golden_dataset(path: Path) -> list[dict]:
    """Load golden dataset."""
    with open(path) as f:
        return json.load(f)


def find_matching_chunks(
    chunks: list[dict],
    docs: dict[str, dict],
    filters: dict,
) -> list[dict]:
    """Find all chunks matching the filter criteria."""
    matching = []

    for chunk in chunks:
        doc_id = chunk.get("document_id")
        if doc_id not in docs:
            continue

        doc = docs[doc_id]

        # Check all filter criteria
        if filters.get("ticker") and doc.get("ticker") != filters["ticker"]:
            continue
        if filters.get("form_type") and doc.get("form_type") != filters["form_type"]:
            continue
        if filters.get("fiscal_period") and doc.get("fiscal_period") != filters["fiscal_period"]:
            continue

        matching.append(chunk)

    return matching


def calculate_word_overlap(reference: str, chunk_content: str) -> float:
    """Calculate percentage of reference words found in chunk."""
    ref_words = set(reference.lower().split())
    chunk_words = set(chunk_content.lower().split())

    if not ref_words:
        return 0.0

    overlap = len(ref_words & chunk_words)
    return overlap / len(ref_words)


def is_substantive_reference(reference: str) -> bool:
    """Check if reference is actual facts, not just a rephrased question."""
    # Red flags: looks like a question
    if reference.endswith("?"):
        return False

    # Should be relatively long (facts, not tiny snippet)
    if len(reference) < 60:
        return False

    # Should have some factual indicators
    has_facts = (
        any(c.isdigit() for c in reference) or
        any(word in reference.lower() for word in [
            "increase", "decrease", "growth", "revenue", "product",
            "segment", "risk", "market", "service", "client", "customer",
        ])
    )

    return has_facts


def validate_item(
    item: dict,
    item_index: int,
    chunks: list[dict],
    docs: dict[str, dict],
) -> ValidationResult:
    """Validate a single golden dataset item."""
    question = item.get("question", "")
    reference = item.get("reference", "")
    filters = item.get("filters", {})

    issues = []

    # Find matching chunks
    matching_chunks = find_matching_chunks(chunks, docs, filters)

    if not matching_chunks:
        issues.append(f"No chunks found matching filters: {filters}")
        return ValidationResult(
            item_index=item_index,
            question=question,
            reference=reference,
            filters=filters,
            found_in_chunks=False,
            matching_chunks_count=0,
            word_overlap=0.0,
            is_substantive=is_substantive_reference(reference),
            issues=issues,
        )

    # Check if reference is in any matching chunk
    best_overlap = 0.0
    found = False

    for chunk in matching_chunks:
        content = chunk.get("content", "")
        overlap = calculate_word_overlap(reference, content)

        if overlap >= 0.7:  # 70%+ threshold
            found = True
            best_overlap = overlap
            break

        best_overlap = max(best_overlap, overlap)

    if not found:
        if best_overlap > 0.5:
            issues.append(f"Reference partially found (overlap={best_overlap:.1%}), not 70%+ match")
        else:
            issues.append(f"Reference not found in {len(matching_chunks)} matching chunks (best overlap={best_overlap:.1%})")

    # Check if substantive
    substantive = is_substantive_reference(reference)
    if not substantive:
        issues.append("Reference may not be substantive (too short or looks like a question)")

    # Check question quality
    if len(question) < 20:
        issues.append("Question is very short (may be ambiguous)")

    return ValidationResult(
        item_index=item_index,
        question=question,
        reference=reference,
        filters=filters,
        found_in_chunks=found,
        matching_chunks_count=len(matching_chunks),
        word_overlap=best_overlap,
        is_substantive=substantive,
        issues=issues,
    )


def print_summary(results: list[ValidationResult]) -> None:
    """Print validation summary."""
    total = len(results)
    valid = sum(1 for r in results if r.found_in_chunks and r.is_substantive and not r.issues)
    found = sum(1 for r in results if r.found_in_chunks)
    substantive = sum(1 for r in results if r.is_substantive)
    with_issues = sum(1 for r in results if r.issues)

    print("\n" + "=" * 70)
    print("GOLDEN DATASET VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Total items                    {total}")
    print(f"  ✓ Fully valid (found + factual) {valid}/{total} ({100*valid//total}%)")
    print(f"  ✓ References found in chunks    {found}/{total} ({100*found//total}%)")
    print(f"  ✓ Substantive (factual)         {substantive}/{total} ({100*substantive//total}%)")
    print(f"  ✗ Items with issues             {with_issues}/{total}")
    print("=" * 70 + "\n")

    # Show problematic items
    if with_issues > 0:
        print("ITEMS WITH ISSUES:\n")
        for r in results:
            if r.issues:
                print(f"[Item {r.item_index}] {r.question[:60]}...")
                print(f"  Reference: {r.reference[:80]}...")
                for issue in r.issues:
                    print(f"  ✗ {issue}")
                print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Validate golden dataset against actual chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", type=Path, default=Path("src/eval/golden_dataset.json"),
        help="Path to the golden dataset JSON.",
    )
    p.add_argument(
        "--chunks-dir", type=Path, default=Path("data/chunks"),
        help="Directory containing chunks.jsonl and documents.jsonl.",
    )
    args = p.parse_args(argv)

    chunks_path = args.chunks_dir / "chunks.jsonl"
    docs_path = args.chunks_dir / "documents.jsonl"

    if not chunks_path.exists():
        log.error("Chunks file not found: %s", chunks_path)
        return 1
    if not docs_path.exists():
        log.error("Documents file not found: %s", docs_path)
        return 1
    if not args.dataset.exists():
        log.error("Golden dataset not found: %s", args.dataset)
        return 1

    log.info("Loading data...")
    chunks = load_chunks(chunks_path)
    docs = load_documents(docs_path)
    dataset = load_golden_dataset(args.dataset)

    log.info("Loaded %d chunks, %d documents, %d golden items", len(chunks), len(docs), len(dataset))

    log.info("Validating %d items...", len(dataset))
    results = []
    for i, item in enumerate(dataset):
        result = validate_item(item, i, chunks, docs)
        results.append(result)

    print_summary(results)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
