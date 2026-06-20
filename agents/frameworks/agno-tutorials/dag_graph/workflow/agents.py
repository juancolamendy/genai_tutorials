"""
doc_pipeline/agents.py
────────────────────────────────────────────────────────────────────────────
Domain-specific LLM agents for the document processing pipeline.

Three states require LLM assistance:
  VALIDATE     — check schema correctness and content quality
  ENRICH       — add tags, summary, and metadata
  HUMAN_REVIEW — simulate an expert reviewer's approval decision

All agents are created via lib.agents.make_agent so they are cached and
reuse the same Claude client across the process.
"""

from pydantic import BaseModel, Field

from engine.agent import make_agent


# structures
# ── Output schemas ────────────────────────────────────────────────────────────
class ValidationResult(BaseModel):
    is_valid:        bool
    schema_version:  str  = ""
    issues:          list[str] = []
    sanitized_data:  dict = {}     # cleaned payload to pass downstream


class EnrichmentResult(BaseModel):
    tags:       list[str]        = []
    summary:    str              = ""
    word_count: int              = 0
    language:   str              = "en"
    metadata:   dict[str, str]  = {}


class ReviewDecision(BaseModel):
    approved:     bool
    reviewer_note: str = ""
    fixed_data:   dict = {}    # corrected payload after human review


# variables
# ── Agents ────────────────────────────────────────────────────────────────────
VALIDATE_AGENT = make_agent(
    name="ValidateAgent",
    description="Validates document schema and content quality.",
    output_schema=ValidationResult,
    instructions=[
        "You receive a raw document payload in <raw_data>.",
        "Check that it has a non-empty 'content' field and a 'schema_version' field.",
        "Set is_valid=true only when both fields pass.",
        "Set is_valid=false and list specific issues when the document fails.",
        "Return the cleaned payload in sanitized_data (trim whitespace, normalise keys).",
        "Never invent content — only clean what is provided.",
    ],
)

ENRICH_AGENT = make_agent(
    name="EnrichAgent",
    description="Enriches a validated document with tags, summary, and metadata.",
    output_schema=EnrichmentResult,
    instructions=[
        "You receive a validated document payload in <validated_data>.",
        "Assign 2–5 relevant topic tags.",
        "Write a concise one-paragraph summary (≤ 60 words).",
        "Count approximate word count of the content field.",
        "Detect the language (ISO 639-1 code).",
        "Add any useful metadata key/value pairs (author, date, topic area).",
    ],
)

REVIEW_AGENT = make_agent(
    name="ReviewAgent",
    description="Simulates a human expert reviewing a flagged document.",
    output_schema=ReviewDecision,
    instructions=[
        "You are simulating an expert document reviewer.",
        "You receive a raw document payload in <raw_data> that failed automatic validation.",
        "Decide whether to approve it (approved=true) or reject it (approved=false).",
        "If approving: provide the corrected payload in fixed_data with at minimum",
        "  'content', 'schema_version' (set to '2.0'), and '_human_approved': true.",
        "If rejecting: explain clearly in reviewer_note and return fixed_data={}.",
        "Approve if the document has substantive content despite minor schema issues.",
        "Reject only if the document is empty, malicious, or completely malformed.",
    ],
)