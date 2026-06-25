#!/usr/bin/env python3
"""Evaluate the quality of the retriever using Ragas LLMContextRecall.

This script evaluates the retriever in isolation — no generator is involved.
For each question it runs Retriever.retrieve(), then asks Ragas whether the
retrieved chunks contain enough information to answer the question.

Metrics
-------
  LLMContextRecall (Coverage)
    For each sentence in the reference answer, the evaluator LLM checks
    whether at least one retrieved chunk supports that sentence.
    Score = supported sentences / total sentences in reference.
    Range 0-1. Higher is better.
    Answers: "Did retriever find passages with the answer?"

  LLMContextPrecision (Relevance)
    For each retrieved chunk, the evaluator LLM checks if it contains
    information relevant to the question.
    Score = relevant chunks / total retrieved chunks.
    Range 0-1. Higher is better.
    Answers: "Are retrieved passages relevant or are they noise?"

Together: "Is the hybrid retriever (vector + FTS + RRF) surfacing
the right passages with high signal?"

Dataset
-------
  golden_dataset_retriever.json — questions with known reference answers
  and optional metadata filters (ticker, form_type, fiscal_period).
  These are factual, section-specific questions where the answer lives
  in a well-defined part of a specific filing.

Environment variables
---------------------
  ANTHROPIC_API_KEY   required
  DATABASE_URL        required
  EMBEDDING_PROVIDER  optional (default: ollama)

Usage
-----
  uv run python eval_retriever.py
  uv run python eval_retriever.py --top-n 10   # retrieve more chunks
  uv run python eval_retriever.py --dataset golden_dataset_retriever.json
  uv run python eval_retriever.py --golden-only  # dry-run, no DB needed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from langchain_anthropic import ChatAnthropic

# Set up path before importing from parent package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from embeddings import OllamaEmbeddings

# Workaround for Ragas import issue with langchain_community vertexai
import types

# Pre-emptively mock the problematic vertexai import
class FakeChatVertexAI:
    pass

vertexai_module = types.ModuleType("vertexai")
vertexai_module.ChatVertexAI = FakeChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = vertexai_module

chat_models = types.ModuleType("chat_models")
chat_models.vertexai = vertexai_module
chat_models.ChatVertexAI = FakeChatVertexAI
sys.modules["langchain_community.chat_models"] = chat_models

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
# Handle different Ragas versions — prefer new collection API, suppress deprecation warnings
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        from ragas.metrics.collections import LLMContextRecall, ContextPrecision as LLMContextPrecision
    except ImportError:
        try:
            from ragas.metrics import LLMContextRecall, LLMContextPrecision
        except ImportError:
            from ragas.metrics import LLMContextRecall
            from ragas.metrics import _ContextPrecision as LLMContextPrecision

from retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_retriever")

EVAL_LLM    = "claude-haiku-4-5-20251001"
DEFAULT_DATASET = Path(__file__).parent / "golden_dataset.json"


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def load_dataset(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path} must be a non-empty JSON array.")
    for i, item in enumerate(items):
        for field in ("question", "reference", "filters"):
            if field not in item:
                raise ValueError(f"Item {i} missing required field '{field}'.")
    log.info("Loaded %d item(s) from %s.", len(items), path)
    return items


# --------------------------------------------------------------------------- #
# Build samples — retriever only, no generator
# --------------------------------------------------------------------------- #
def build_samples(
    dataset: list[dict],
    retriever: Retriever,
    top_k: int,
    top_n: int,
) -> list[SingleTurnSample]:
    """Retrieve chunks for each question. No response field — not needed for
    LLMContextRecall, which only looks at retrieved_contexts vs reference."""
    samples: list[SingleTurnSample] = []
    total = len(dataset)

    for i, item in enumerate(dataset, start=1):
        question  = item["question"]
        reference = item["reference"]
        filters   = item.get("filters", {})

        log.info("[%d/%d] %s", i, total, question[:70])

        results  = retriever.retrieve(question, top_k=top_k, top_n=top_n, **filters)
        contexts = [r.payload["content"] for r in results]

        if not contexts:
            log.warning("  No chunks retrieved for: %s", question[:60])

        # Note: no response field.
        # LLMContextRecall only requires user_input, retrieved_contexts,
        # and reference — it does not need a generated answer.
        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            reference=reference,
        ))

    return samples


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def print_summary(result, n_samples: int) -> None:
    df = result.to_pandas()
    metric_cols = [
        c for c in df.columns
        if c not in ("user_input", "retrieved_contexts", "response", "reference")
    ]
    print("\n" + "=" * 60)
    print("RETRIEVER EVALUATION SUMMARY")
    print("=" * 60)
    for col in metric_cols:
        mean  = df[col].mean()
        worst = df[col].min()
        best  = df[col].max()
        print(f"  {col:<28}  mean={mean:.4f}  min={worst:.4f}  max={best:.4f}")
    print("─" * 60)
    print(f"  {'Samples evaluated':<28}  {n_samples}")
    print("=" * 60 + "\n")


def save_results(result, output_path: Path) -> None:
    """Disabled: results now output only to terminal."""
    pass


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate retriever quality with Ragas LLMContextRecall.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                   help="Path to the golden retriever dataset JSON.")
    p.add_argument("--top-n", type=int, default=5,
                   help="Number of chunks to retrieve per question.")
    p.add_argument("--eval-model", default=EVAL_LLM,
                   help="Claude model used as the Ragas evaluator LLM.")
    p.add_argument("--top-k", type=int, default=40,
                   help="Candidate pool size per retriever before fusion.")
    p.add_argument("--golden-only", action="store_true",
                   help="Dry-run: skip retrieval, no DB needed.")
    p.add_argument("--database-url",   default=os.environ.get("DATABASE_URL", ""))
    p.add_argument("--anthropic-api-key", default=os.environ.get("ANTHROPIC_API_KEY", ""))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.anthropic_api_key:
        log.error("ANTHROPIC_API_KEY is not set.")
        return 1
    if not args.golden_only and not args.database_url:
        log.error("DATABASE_URL is not set. Use --golden-only to skip the DB.")
        return 1

    try:
        dataset = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    evaluator_llm = LangchainLLMWrapper(
        ChatAnthropic(model=args.eval_model, api_key=args.anthropic_api_key, temperature=0)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings()
    )

    if args.golden_only:
        log.info("--golden-only: dry-run, no retrieval.")
        samples = [
            SingleTurnSample(
                user_input=item["question"],
                retrieved_contexts=["(dry-run — no retrieval performed)"],
                reference=item["reference"],
            )
            for item in dataset
        ]
    else:
        with Retriever.from_credentials(args.database_url) as r:
            samples = build_samples(dataset, r, args.top_k, args.top_n)

    if not samples:
        log.error("No samples produced.")
        return 1

    log.info("Evaluating %d sample(s) …", len(samples))

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            LLMContextRecall(),      # Coverage: did retriever find relevant passages?
            LLMContextPrecision(),   # Relevance: were retrieved passages relevant?
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    print_summary(result, len(samples))
    return 0


if __name__ == "__main__":
    sys.exit(main())