"""Domain-specific LLM chains for the document processing pipeline.

Three states require LLM assistance:
  VALIDATE     — check schema correctness and content quality
  ENRICH       — add tags, summary, and metadata
  HUMAN_REVIEW — simulate an expert reviewer's approval decision

All chains are created via engine.chain.make_chain so they are cached and
reuse the same Claude client across the process.
"""

from pydantic import BaseModel
from src.engine.chain import make_chain


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ValidationResult(BaseModel):
    """Result of document validation."""

    is_valid: bool
    schema_version: str = ""
    issues: list[str] = []
    sanitized_data: dict = {}


class EnrichmentResult(BaseModel):
    """Result of document enrichment."""

    tags: list[str] = []
    summary: str = ""
    word_count: int = 0
    language: str = "en"
    metadata: dict[str, str] = {}


class ReviewDecision(BaseModel):
    """Result of human review."""

    approved: bool
    reviewer_note: str = ""
    fixed_data: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# CHAINS
# ─────────────────────────────────────────────────────────────────────────────

VALIDATE_CHAIN = make_chain(
    name="ValidateChain",
    description="Validates document schema and content quality.",
    system_prompt="""You are a document validator. Analyze raw document payloads and determine validity.

Your job is to:
1. Check that the document has non-empty 'content' field
2. Check for 'schema_version' field presence
3. Return cleaned/sanitized data with normalized keys and trimmed whitespace
4. List any issues found

Respond ONLY with valid JSON matching this structure:
{{
  "is_valid": <bool>,
  "schema_version": <str>,
  "issues": [<str>, ...],
  "sanitized_data": {{...}}
}}

Set is_valid=true only when both 'content' and 'schema_version' fields are valid.
Never invent content — only clean what is provided.""",
    output_schema=ValidationResult,
)

ENRICH_CHAIN = make_chain(
    name="EnrichChain",
    description="Enriches a validated document with tags, summary, and metadata.",
    system_prompt="""You are a document enrichment specialist. Add value to validated documents.

Your job is to:
1. Assign 2–5 relevant topic tags
2. Write a concise one-paragraph summary (≤ 60 words)
3. Estimate the word count of the content
4. Detect the language (ISO 639-1 code)
5. Add useful metadata key/value pairs (author, date, topic area, etc)

Respond ONLY with valid JSON matching this structure:
{{
  "tags": [<str>, ...],
  "summary": <str>,
  "word_count": <int>,
  "language": <str>,
  "metadata": {{<str>: <str>}}
}}""",
    output_schema=EnrichmentResult,
)

REVIEW_CHAIN = make_chain(
    name="ReviewChain",
    description="Simulates a human expert reviewing a flagged document.",
    system_prompt="""You are a document review expert. Decide whether to approve flagged documents.

You receive raw documents that failed automatic validation. Your job is to:
1. Decide whether to approve (true) or reject (false)
2. If approving: provide corrected payload with 'content', 'schema_version' (set to '2.0'), '_human_approved': true
3. If rejecting: explain clearly in reviewer_note and return fixed_data={{}}

Approve if document has substantive content despite minor schema issues.
Reject only if empty, malicious, or completely malformed.

Respond ONLY with valid JSON matching this structure:
{{
  "approved": <bool>,
  "reviewer_note": <str>,
  "fixed_data": {{...}}
}}""",
    output_schema=ReviewDecision,
)


__all__ = [
    "ValidationResult",
    "EnrichmentResult",
    "ReviewDecision",
    "VALIDATE_CHAIN",
    "ENRICH_CHAIN",
    "REVIEW_CHAIN",
]
