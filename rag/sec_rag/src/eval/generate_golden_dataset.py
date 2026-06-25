#!/usr/bin/env python3
"""Generate and validate golden dataset from actual chunks.

Reads chunks.jsonl and documents.jsonl to extract factual questions and references
that are guaranteed to exist in the database. Validates that each reference
can actually be retrieved, ensuring dataset quality.

Usage
-----
    uv run python generate_golden_dataset.py
    uv run python generate_golden_dataset.py --output custom_golden_dataset.json
    uv run python generate_golden_dataset.py --min-chunk-length 200 --target-items 50
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_golden_dataset")


@dataclass
class Chunk:
    """Chunk record from chunks.jsonl"""
    id: str
    document_id: str
    section: str | None
    chunk_index: int
    content: str
    ticker: str | None = None
    company: str | None = None
    form_type: str | None = None
    fiscal_period: str | None = None
    url: str | None = None


@dataclass
class Document:
    """Document record from documents.jsonl"""
    id: str
    ticker: str
    company: str
    form_type: str
    fiscal_period: str
    filing_date: str | None = None
    url: str | None = None


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_chunks(path: Path) -> list[Chunk]:
    """Load chunks from JSONL."""
    chunks = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            chunks.append(Chunk(
                id=data["id"],
                document_id=data["document_id"],
                section=data.get("section"),
                chunk_index=data["chunk_index"],
                content=data["content"],
            ))
    log.info("Loaded %d chunks.", len(chunks))
    return chunks


def load_documents(path: Path) -> dict[str, Document]:
    """Load documents from JSONL and index by ID."""
    docs = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            doc = Document(
                id=data["id"],
                ticker=data.get("ticker"),
                company=data.get("company"),
                form_type=data.get("form_type"),
                fiscal_period=data.get("fiscal_period"),
                filing_date=data.get("filing_date"),
                url=data.get("url"),
            )
            docs[doc.id] = doc
    log.info("Loaded %d documents.", len(docs))
    return docs


# --------------------------------------------------------------------------- #
# Question and reference extraction
# --------------------------------------------------------------------------- #
def extract_key_sentences(text: str, max_length: int = 300) -> list[str]:
    """Extract key sentences (2-3 sentences) from chunk content.

    Returns sentences that:
    - Are between 80-300 chars (substantial, meaningful content)
    - Contain numerical data, comparisons, or specific facts
    - Exclude boilerplate (addresses, stock symbols, URLs)
    """
    # Skip boilerplate cover page markers
    boilerplate_markers = [
        'zip code', 'principal executive', 'telephone number',
        'common stock', 'par value', 'registrant', 'securities registered',
        'commission file number', 'industry code',
        'http', '@', '.com', '.org', '(address)', '= (',
    ]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    good_sentences = []
    for sent in sentences:
        # Skip boilerplate
        if any(marker in sent.lower() for marker in boilerplate_markers):
            continue

        if 80 < len(sent) < max_length:
            # Prefer sentences with facts, figures, or business content
            if any(c.isdigit() for c in sent) or \
               'increase' in sent.lower() or \
               'decrease' in sent.lower() or \
               'risk' in sent.lower() or \
               'primary' in sent.lower() or \
               'largest' in sent.lower() or \
               'growth' in sent.lower() or \
               'revenue' in sent.lower() or \
               'segment' in sent.lower() or \
               'service' in sent.lower() or \
               'product' in sent.lower() or \
               'compete' in sent.lower() or \
               'market' in sent.lower():
                good_sentences.append(sent)

    return good_sentences[:3]  # Up to 3 sentences


def generate_questions(chunk: Chunk, reference: str) -> list[tuple[str, str]]:
    """Generate meaningful question-reference pairs from a chunk.

    Returns list of (question, reference) tuples based on content.
    """
    questions = []
    ref_lower = reference.lower()
    company = chunk.company or "the company"

    # Pattern 1: Risk disclosure
    if "risk" in ref_lower and len(reference) > 100:
        questions.append((
            f"What are {company}'s main risk factors according to the 10-K?",
            reference
        ))

    # Pattern 2: Revenue or business drivers
    if any(word in ref_lower for word in ["revenue", "sales", "income", "grow", "increase"]):
        if "%" in reference or any(c.isdigit() for c in reference):
            questions.append((
                f"What drove {company}'s financial performance?",
                reference
            ))
        else:
            questions.append((
                f"What are {company}'s key business drivers?",
                reference
            ))

    # Pattern 3: Product/segment-specific
    if any(word in ref_lower for word in ["segment", "product", "service", "cloud", "aws", "iphone", "azure"]):
        questions.append((
            f"What are {company}'s primary product segments?",
            reference
        ))

    # Pattern 4: Competitive landscape or market position
    if any(word in ref_lower for word in ["compet", "market", "market share", "advantage", "leadership"]):
        questions.append((
            f"How does {company} describe its competitive position?",
            reference
        ))

    # Pattern 5: Specific numbers or metrics
    if any(c.isdigit() for c in reference) and len(reference) > 80:
        questions.append((
            f"What specific financial metrics does {company} disclose?",
            reference
        ))

    # Only return non-empty questions
    return [q for q in questions if q[0].strip()]


# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #
def is_boilerplate_chunk(content: str) -> bool:
    """Check if chunk is pure form/boilerplate content (not business substance)."""
    content_lower = content.lower()

    # Form/structural markers
    structural_markers = [
        'securities exchange act',
        'large accelerated filer',
        'smaller reporting company',
        'form 10-k',
        'sec.gov',
        'rule 12b-2',
        'transition report',
        'exchange act of 1934',
        'mark one',
        'registrant phone',
        'item ',  # Section headers like "Item 1.", "Item 7."
        'management\'s discussion',
        'changes in and disagreements',
    ]

    # Count structural markers
    structural_count = sum(1 for marker in structural_markers
                          if marker in content_lower)

    # If mostly structural, skip
    if structural_count >= 2:
        return True

    # Also skip if very little actual text content (high ratio of whitespace/formatting)
    lines = content.split('\n')
    meaningful_lines = [l for l in lines if l.strip() and len(l.strip()) > 20]
    if len(meaningful_lines) < 3:
        return True

    return False


def generate_dataset(
    chunks: list[Chunk],
    docs: dict[str, Document],
    target_items: int = 50,
    min_chunk_length: int = 200,
) -> list[dict]:
    """Generate golden dataset from chunks.

    Strategy:
    1. Filter chunks by minimum length (substantial content)
    2. Skip boilerplate/form-only chunks
    3. Extract key sentences as references
    4. Generate questions from patterns
    5. Distribute across companies and time periods
    6. Target N items with high diversity
    """
    candidates = []

    # Enrich chunks with document metadata
    for chunk in chunks:
        if chunk.document_id in docs:
            doc = docs[chunk.document_id]
            chunk.ticker = doc.ticker
            chunk.company = doc.company
            chunk.form_type = doc.form_type
            chunk.fiscal_period = doc.fiscal_period
            chunk.url = doc.url

    # Filter chunks by length and remove boilerplate
    long_chunks = [c for c in chunks if len(c.content) >= min_chunk_length]
    business_chunks = [c for c in long_chunks if not is_boilerplate_chunk(c.content)]
    log.info("Filtered to %d chunks with length >= %d chars.", len(long_chunks), min_chunk_length)
    log.info("Removed %d boilerplate chunks, keeping %d business-focused chunks.",
             len(long_chunks) - len(business_chunks), len(business_chunks))

    # Extract questions and references
    for chunk in business_chunks:
        sentences = extract_key_sentences(chunk.content)
        if not sentences:
            continue

        reference = " ".join(sentences)
        questions = generate_questions(chunk, reference)

        for question, ref in questions:
            candidates.append({
                "question": question,
                "reference": ref,
                "filters": {
                    "ticker": chunk.ticker,
                    "form_type": chunk.form_type,
                    "fiscal_period": chunk.fiscal_period,
                },
                "source_chunk_id": chunk.id,
                "source_section": chunk.section or "(cover page)",
            })

    log.info("Generated %d candidate Q&A pairs.", len(candidates))

    # Diversify by company and period
    by_company = {}
    for item in candidates:
        ticker = item["filters"]["ticker"]
        if ticker not in by_company:
            by_company[ticker] = []
        by_company[ticker].append(item)

    # Select items to balance across companies
    selected = []
    items_per_company = target_items // len(by_company)

    for ticker, items in by_company.items():
        # Diversify within company
        selected.extend(items[:items_per_company])

    # Fill remaining slots
    remaining = target_items - len(selected)
    if remaining > 0:
        not_selected = [i for i in candidates if i not in selected]
        selected.extend(not_selected[:remaining])

    log.info("Selected %d items for final dataset.", len(selected))

    # Remove source info (only needed for validation)
    for item in selected:
        del item["source_chunk_id"]
        del item["source_section"]

    return selected


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_dataset(
    dataset: list[dict],
    chunks: list[Chunk],
) -> dict[str, int]:
    """Validate that each reference exists in chunks.

    Returns statistics on validation.
    """
    stats = {
        "total": len(dataset),
        "valid": 0,
        "missing": 0,
        "partial": 0,
    }

    chunk_texts = {c.id: c.content for c in chunks}

    for item in dataset:
        reference = item["reference"].lower()
        filters = item["filters"]

        # Find chunks matching filters
        matching_chunks = [
            c for c in chunks
            if c.ticker == filters["ticker"]
            and c.form_type == filters.get("form_type")
            and c.fiscal_period == filters.get("fiscal_period")
        ]

        if not matching_chunks:
            log.warning("No chunks found for: %s", item["question"][:60])
            stats["missing"] += 1
            continue

        # Check if reference is in any matching chunk
        found = False
        for chunk in matching_chunks:
            # Check if key phrases from reference exist in chunk
            ref_words = set(reference.split())
            chunk_words = set(chunk.content.lower().split())

            # If 70%+ of reference words are in chunk, consider it valid
            overlap = len(ref_words & chunk_words) / len(ref_words)
            if overlap >= 0.7:
                found = True
                stats["valid"] += 1
                break

        if not found:
            log.warning("Reference not fully in chunks: %s", item["question"][:60])
            stats["partial"] += 1

    return stats


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def save_dataset(dataset: list[dict], path: Path) -> None:
    """Save dataset to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(dataset, f, indent=2)
    log.info("Saved %d items to %s", len(dataset), path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate golden dataset from actual chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--chunks-dir", type=Path, default=Path("data/chunks"),
        help="Directory containing chunks.jsonl and documents.jsonl.",
    )
    p.add_argument(
        "--output", type=Path, default=Path("src/eval/golden_dataset.json"),
        help="Path to write generated golden dataset.",
    )
    p.add_argument(
        "--target-items", type=int, default=50,
        help="Target number of items in final dataset.",
    )
    p.add_argument(
        "--min-chunk-length", type=int, default=200,
        help="Minimum chunk length to extract from.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    chunks_path = args.chunks_dir / "chunks.jsonl"
    docs_path = args.chunks_dir / "documents.jsonl"

    if not chunks_path.exists():
        log.error("Chunks file not found: %s", chunks_path)
        return 1
    if not docs_path.exists():
        log.error("Documents file not found: %s", docs_path)
        return 1

    # Load data
    chunks = load_chunks(chunks_path)
    docs = load_documents(docs_path)

    # Generate dataset
    dataset = generate_dataset(
        chunks, docs,
        target_items=args.target_items,
        min_chunk_length=args.min_chunk_length,
    )

    # Validate
    log.info("Validating dataset...")
    stats = validate_dataset(dataset, chunks)
    log.info(
        "Validation: %d valid, %d partial, %d missing out of %d total",
        stats["valid"], stats["partial"], stats["missing"], stats["total"]
    )

    # Save
    save_dataset(dataset, args.output)

    print("\n" + "=" * 60)
    print(f"✓ Golden dataset generated: {args.output}")
    print(f"  Items: {len(dataset)}")
    print(f"  Valid: {stats['valid']}/{stats['total']}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
