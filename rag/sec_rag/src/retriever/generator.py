#!/usr/bin/env python3
"""Answer generator for the SEC RAG pipeline.

Takes a user question and a list of retrieved chunks from retriever.py,
calls the Anthropic Claude API with a strict grounded-answer prompt,
and returns a structured GeneratorResponse.

This is the generation stage of the pipeline:

    retriever.py  →  list[RankedResult]
                           ↓
    generator.py  →  GeneratorResponse
                           ↓
    eval_rag.py   →  SingleTurnSample (for Ragas)

The generator is intentionally strict: it instructs the LLM to answer ONLY
from the provided context and to say "Insufficient context." when the chunks
do not contain enough information.  This makes Faithfulness scores meaningful
— a high score means the answer is genuinely grounded, not that the LLM fell
back on parametric knowledge.

Environment variables
---------------------
  ANTHROPIC_API_KEY   required
  DATABASE_URL     required (for retrieval)

Usage (library)
---------------
    from retriever import Retriever
    from generator import Generator

    with Retriever.from_env() as r:
        results = r.retrieve("What are Apple's supply chain risks?", ticker="AAPL")

    g = Generator.from_env()
    resp = g.generate("What are Apple's supply chain risks?", results)
    print(resp.answer)
    print(resp.contexts)   # the text passages fed to the LLM

Usage (CLI)
-----------
    uv run python -m src.retriever.generator "What are Apple's supply chain risks?" \\
        --ticker AAPL --form-type 10-K
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import textwrap
from dataclasses import dataclass, field

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    # Running as module
    from .retriever import RankedResult, Retriever
except ImportError:
    # Running as script directly
    from sys import path
    path.insert(0, os.path.dirname(__file__))
    from retriever import RankedResult, Retriever  # type: ignore

log = logging.getLogger("generator")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

CHAT_MODEL    = "claude-haiku-4-5-20251001"
DEFAULT_TOP_N = 5

# The system prompt enforces grounding. The LLM must not use knowledge
# outside the provided passages — this is what makes Faithfulness testable.
_SYSTEM_PROMPT = """\
# ROLE:
You are a financial analyst assistant specialising in SEC filings.
# GOAL:
Your goal is to answer questions using ONLY the context passages provided.
Follow the instructions below.
# INSTRUCTIONS:
- Be concise, accurate, and cite the section when relevant.
- If the context does not contain enough information to answer, respond with
  exactly: "Insufficient context."
- Do not use any knowledge outside the provided passages.
"""

_USER_PROMPT = """\
Context passages:
{context}

Question: {question}

Answer:\
"""


# --------------------------------------------------------------------------- #
# Response dataclass
# --------------------------------------------------------------------------- #
@dataclass
class GeneratorResponse:
    """Structured output from the generator."""
    question:  str
    answer:    str
    contexts:  list[str]          = field(default_factory=list)
    # Source metadata forwarded from retriever results for traceability.
    sources:   list[dict]         = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Chain creation
# --------------------------------------------------------------------------- #
def create_chain(model: str = CHAT_MODEL, temperature: float = 0.0):
    """Create a LangChain chain for grounded answer generation.

    Uses ChatAnthropic (Claude) with built-in retry logic and exponential backoff.
    Includes StrOutputParser to extract answer string directly.

    Parameters
    ----------
    model:
        Anthropic Claude model name (default: claude-3-5-haiku-20241022).
    temperature:
        Sampling temperature (0 = deterministic, for evaluation).

    Returns
    -------
    A LangChain Runnable that takes {"context": str, "question": str}
    and returns the answer string directly.
    """
    # LangChain's ChatAnthropic handles retries internally
    llm = ChatAnthropic(
        model=model,
        temperature=temperature,
        # max_retries defaults to 2; we can override if needed
    )

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("user", _USER_PROMPT),
    ])

    # Create chain: prompt → llm → string parser
    # StrOutputParser extracts content from AIMessage automatically
    chain = prompt | llm | StrOutputParser()

    return chain


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class Generator:
    """Grounded answer generator backed by LangChain + Anthropic Claude."""

    def __init__(self, chain, model: str = CHAT_MODEL) -> None:
        self._chain = chain
        self._model = model
        self._answer_cache: dict[str, str] = {}  # (question, context_hash) → answer

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, model: str = CHAT_MODEL) -> "Generator":
        """Construct from ANTHROPIC_API_KEY environment variable."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        return cls.from_credentials(api_key, model=model)

    @classmethod
    def from_credentials(cls, anthropic_api_key: str, model: str = CHAT_MODEL) -> "Generator":
        # LangChain's ChatAnthropic reads ANTHROPIC_API_KEY from environment or explicit init
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
        chain = create_chain(model=model)
        return cls(chain, model=model)

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(
        self,
        question: str,
        results: list[RankedResult],
    ) -> GeneratorResponse:
        """Generate a grounded answer from retriever results.

        Parameters
        ----------
        question:
            The user's original question.
        results:
            Ranked chunks from Retriever.retrieve(). The content of each
            chunk is assembled into the context block sent to the LLM.

        Returns
        -------
        GeneratorResponse with question, answer, contexts, and sources.

        Raises
        ------
        ValueError
            If question is empty/whitespace-only.
        """
        # Validate question
        question_clean = question.strip()
        if not question_clean:
            raise ValueError("Question cannot be empty or whitespace-only.")

        # Handle no results case
        if not results:
            log.warning("No retrieved chunks — returning insufficient context.")
            return GeneratorResponse(
                question=question_clean,
                answer="Insufficient context.",
                contexts=[],
                sources=[],
            )

        # Validate and extract contexts
        contexts = [r.payload["content"] for r in results]
        if not contexts:
            log.warning("No contexts found in results.")
            return GeneratorResponse(
                question=question_clean,
                answer="Insufficient context.",
                contexts=[],
                sources=[],
            )

        # Build sources with document_id
        sources = [
            {
                "document_id":  r.payload.get("document_id"),
                "chunk_id":     r.chunk_id,
                "rrf_score":    r.rrf_score,
                "section":      r.payload.get("section") or "(none)",
                "ticker":       r.payload.get("ticker"),
                "fiscal_period": r.payload.get("fiscal_period"),
                "url":          r.payload.get("url"),
            }
            for r in results
        ]

        # Build context block
        context_block = "\n\n---\n\n".join(
            f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)
        )

        # Check cache
        cache_key = _make_cache_key(question_clean, context_block)
        if cache_key in self._answer_cache:
            log.debug("Answer cache hit for question: %r", question_clean[:50])
            answer = self._answer_cache[cache_key]
        else:
            # Generate answer using chain
            log.info(
                "Generating answer for: %s  (%d context chunks, model=%s)",
                question_clean[:60], len(contexts), self._model,
            )

            try:
                # LangChain chain invocation (returns string directly via StrOutputParser)
                answer = self._chain.invoke({
                    "context": context_block,
                    "question": question_clean,
                }).strip()

            except Exception as exc:
                log.error("Generation failed: %s", exc)
                raise

            # Validate answer is non-empty
            if not answer:
                log.warning("Generated answer is empty, returning insufficient context.")
                answer = "Insufficient context."
            else:
                # Cache the answer
                self._answer_cache[cache_key] = answer
                log.info("Answer: %s", answer[:80])

        return GeneratorResponse(
            question=question_clean,
            answer=answer,
            contexts=contexts,
            sources=sources,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_cache_key(question: str, context: str) -> str:
    """Create a cache key from question and context."""
    combined = f"{question}|||{context}"
    return hashlib.sha256(combined.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a grounded answer from SEC filings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("question", help="Question to answer.")
    p.add_argument("--ticker",         default=None)
    p.add_argument("--form-type",      default=None)
    p.add_argument("--fiscal-period",  default=None)
    p.add_argument("--top-n",  type=int, default=DEFAULT_TOP_N,
                   help="Number of chunks to retrieve.")
    p.add_argument("--model",  default=CHAT_MODEL,
                   help="Anthropic Claude chat model.")
    p.add_argument("--database-url",   default=os.environ.get("DATABASE_URL", ""))
    p.add_argument("--anthropic-api-key", default=os.environ.get("ANTHROPIC_API_KEY", ""))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.database_url:
        log.error("DATABASE_URL is not set.")
        return 1
    if not args.anthropic_api_key:
        log.error("ANTHROPIC_API_KEY is not set.")
        return 1

    with Retriever.from_credentials(args.database_url) as r:
        results = r.retrieve(
            args.question,
            ticker=args.ticker,
            form_type=args.form_type,
            fiscal_period=args.fiscal_period,
            top_n=args.top_n,
        )

    g = Generator.from_credentials(args.anthropic_api_key, model=args.model)
    resp = g.generate(args.question, results)

    width = 72
    print(f"\n{'─' * width}")
    print(f"Q: {resp.question}")
    print(f"{'─' * width}")
    print(textwrap.fill(resp.answer, width=width))
    print(f"{'─' * width}")
    print("Sources:")
    for s in resp.sources:
        section_label = s["section"] if s["section"] != "(none)" else "(cover page)"
        doc_id_short = str(s['document_id'])[:8] if s['document_id'] else "unknown"
        print(f"  [{s['ticker']} {s['fiscal_period']}] {section_label} "
              f"(doc_id={doc_id_short}... score={s['rrf_score']:.4f})")
    print(f"{'─' * width}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
