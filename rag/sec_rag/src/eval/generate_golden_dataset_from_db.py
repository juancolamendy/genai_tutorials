#!/usr/bin/env python3
"""Generate golden dataset directly from database chunks.

This ensures the golden dataset references are actual chunks
that exist in the indexed database, not from stale JSONL files.

Usage
-----
    uv run python src/eval/generate_golden_dataset_from_db.py
    uv run python src/eval/generate_golden_dataset_from_db.py --target-items 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from dataclasses import dataclass

import psycopg
from pgvector.psycopg import register_vector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_golden_dataset_from_db")


@dataclass
class DbChunk:
    """Chunk from database."""
    id: str
    content: str
    section: str | None
    ticker: str
    form_type: str
    fiscal_period: str
    company: str


def is_boilerplate_chunk(content: str) -> bool:
    """Check if chunk is pure form/boilerplate content."""
    content_lower = content.lower()

    # Skip if starts with form headers or boilerplate
    boilerplate_starts = [
        'item ', 'securities and exchange commission', 'form 10-',
        'mark one', 'exchange act', 'large accelerated', 'sec.gov',
        'this annual report', 'commission file'
    ]

    for start in boilerplate_starts:
        if content_lower.strip().startswith(start):
            return True

    # Form/structural markers
    structural_markers = [
        'securities exchange act',
        'large accelerated filer',
        'smaller reporting company',
        'sec.gov',
        'rule 12b-2',
        'exchange act of 1934',
        'mark one',
        'registrant phone',
    ]

    structural_count = sum(1 for marker in structural_markers
                          if marker in content_lower)

    if structural_count >= 2:
        return True

    # Skip if mostly numbers/tables
    lines = content.split('\n')
    meaningful_lines = [l for l in lines if l.strip() and len(l.strip()) > 20]
    if len(meaningful_lines) < 3:
        return True

    return False


def extract_key_sentences(text: str, max_length: int = 300) -> list[str]:
    """Extract key sentences from chunk."""
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
        if any(marker in sent.lower() for marker in boilerplate_markers):
            continue

        if 80 < len(sent) < max_length:
            if any(c.isdigit() for c in sent) or \
               any(word in sent.lower() for word in [
                   'increase', 'decrease', 'risk', 'primary', 'largest',
                   'growth', 'revenue', 'segment', 'service', 'product',
                   'compete', 'market', 'challenge', 'business', 'operations'
               ]):
                good_sentences.append(sent)

    return good_sentences[:3]


def calculate_overlap(question: str, reference: str) -> float:
    """Calculate word overlap between question and reference."""
    q_words = set(w.lower() for w in question.split() if len(w) > 3)
    r_words = set(w.lower() for w in reference.lower().split() if len(w) > 3)
    if not q_words:
        return 0.0
    return len(q_words & r_words) / len(q_words)


def generate_questions(chunk: DbChunk, reference: str) -> list[tuple[str, str]]:
    """Generate Q&A pairs from chunk.

    Only generate questions whose key terms are present in the reference.
    Requires minimum 20% word overlap to ensure semantic alignment.
    """
    questions = []
    ref_lower = reference.lower()
    company = chunk.company or "the company"
    candidate_questions = []

    # Risk factors: only if "risk" appears in reference
    if "risk" in ref_lower and len(reference) > 100:
        candidate_questions.append(
            f"What are {company}'s main risk factors according to the 10-K?"
        )

    # Financial performance: only if financial keywords in reference
    if any(word in ref_lower for word in ["revenue", "sales", "income", "earnings", "financial"]):
        if "%" in reference or any(c.isdigit() for c in reference):
            candidate_questions.append(
                f"What drove {company}'s financial performance?"
            )
        else:
            candidate_questions.append(
                f"What are {company}'s key business drivers?"
            )

    # Product/service segments: only if "segment" is in reference
    if "segment" in ref_lower:
        candidate_questions.append(
            f"What are {company}'s primary product segments?"
        )
    elif any(word in ref_lower for word in ["product", "service"]):
        candidate_questions.append(
            f"What {company}'s products and services does the 10-K describe?"
        )

    # Competitive position: only if competition-related words in reference
    if any(word in ref_lower for word in ["compet", "market", "advantage", "leadership"]):
        candidate_questions.append(
            f"How does {company} describe its competitive position?"
        )

    # Financial metrics: only if numbers AND financial context
    if any(c.isdigit() for c in reference) and len(reference) > 80:
        if any(word in ref_lower for word in ["financial", "metric", "billion", "million", "revenue", "income"]):
            candidate_questions.append(
                f"What specific financial metrics does {company} disclose?"
            )

    # Filter to only questions with sufficient overlap with reference
    min_overlap = 0.15  # At least 15% of question words in reference
    for q in candidate_questions:
        if calculate_overlap(q, reference) >= min_overlap:
            questions.append((q, reference))

    return questions


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate golden dataset from actual database chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output", type=Path, default=Path("src/eval/golden_dataset.json"),
        help="Path to write golden dataset.",
    )
    p.add_argument(
        "--target-items", type=int, default=50,
        help="Target number of items.",
    )
    p.add_argument(
        "--database-url", default=os.environ.get("DATABASE_URL", ""),
        help="Database URL.",
    )
    args = p.parse_args(argv)

    if not args.database_url:
        log.error("DATABASE_URL is not set.")
        return 1

    # Connect to database
    conn = psycopg.connect(args.database_url)
    register_vector(conn)
    log.info("Connected to database.")

    # Fetch chunks
    query = """
    SELECT c.id, c.content, c.section, d.ticker, d.form_type, d.fiscal_period, d.company
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE c.content IS NOT NULL
    ORDER BY d.ticker, d.fiscal_period, c.chunk_index;
    """

    with conn.cursor() as cur:
        cur.execute(query)
        all_chunks = []
        for row in cur.fetchall():
            all_chunks.append(DbChunk(
                id=row[0],
                content=row[1],
                section=row[2],
                ticker=row[3],
                form_type=row[4],
                fiscal_period=row[5],
                company=row[6],
            ))

    conn.close()
    log.info("Loaded %d chunks from database.", len(all_chunks))

    # Filter
    long_chunks = [c for c in all_chunks if len(c.content) >= 200]
    business_chunks = [c for c in long_chunks if not is_boilerplate_chunk(c.content)]
    log.info("Filtered to %d business-focused chunks.", len(business_chunks))

    # Generate Q&A pairs
    candidates = []
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
            })

    log.info("Generated %d Q&A candidates.", len(candidates))

    # Balance by company
    by_company = {}
    for item in candidates:
        ticker = item["filters"]["ticker"]
        if ticker not in by_company:
            by_company[ticker] = []
        by_company[ticker].append(item)

    selected = []
    items_per_company = max(1, args.target_items // len(by_company))

    for ticker, items in by_company.items():
        selected.extend(items[:items_per_company])

    remaining = args.target_items - len(selected)
    if remaining > 0:
        not_selected = [i for i in candidates if i not in selected]
        selected.extend(not_selected[:remaining])

    log.info("Selected %d items for final dataset.", len(selected))

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(selected, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✓ Golden dataset generated: {args.output}")
    print(f"  Items: {len(selected)}")
    print(f"  Companies: {len(by_company)}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
