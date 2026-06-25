#!/usr/bin/env python3
"""Evaluate the quality of the full RAG pipeline using Ragas.

This script evaluates the complete pipeline — retriever + generator — using
two generation-focused metrics.  It assumes the retriever is working
adequately (verified separately by eval_retriever.py) and focuses on whether
the generator produces answers that are grounded and factually correct.

Metrics
-------
  Faithfulness
    Are all claims in the generated answer supported by the retrieved chunks?
    Detects hallucination: the LLM saying things not present in the context.
    Score = grounded claims / total claims in response.
    Range 0-1. Higher is better.

  FactualCorrectness
    Does the generated answer match the reference answer?
    Measures end-to-end pipeline quality — wrong answers from bad retrieval
    or bad generation both lower this score.
    Range 0-1. Higher is better.

What each score tells you
-------------------------
  High Faithfulness + High FactualCorrectness  →  pipeline is working well
  Low  Faithfulness + High FactualCorrectness  →  LLM is using parametric
       knowledge instead of context (lucky but not grounded)
  High Faithfulness + Low  FactualCorrectness  →  retriever is returning the
       wrong chunks; generator is faithfully using bad context
  Low  Faithfulness + Low  FactualCorrectness  →  both components need work

Dataset
-------
  golden_dataset_generator.json — questions that require multi-sentence,
  reasoning-style answers. These are harder than the retriever dataset and
  are designed to stress-test whether the generator synthesises context well.

Environment variables
---------------------
  OPENAI_API_KEY   required
  DATABASE_URL     required

Usage
-----
  uv run python eval_generator.py
  uv run python eval_generator.py --dataset golden_dataset_generator.json
  uv run python eval_generator.py --top-n 10
  uv run python eval_generator.py --golden-only  # dry-run, no DB needed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Workaround for Ragas import issue with langchain_community vertexai
import sys
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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from ragas.metrics import FactualCorrectness, Faithfulness

from generator import Generator
from retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_generator")

EVAL_LLM    = "gpt-4o"
EMBED_MODEL = "text-embedding-3-large"
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
# Build samples — full pipeline: retriever → generator
# --------------------------------------------------------------------------- #
def build_samples(
    dataset: list[dict],
    retriever: Retriever,
    generator: Generator,
    top_k: int,
    top_n: int,
) -> list[SingleTurnSample]:
    """Run each question through the full pipeline.

    retrieved_contexts  — the chunks fed to the generator
    response            — what the generator produced from those chunks
    reference           — the known correct answer from the golden dataset

    Faithfulness uses retrieved_contexts + response.
    FactualCorrectness uses response + reference.
    """
    samples: list[SingleTurnSample] = []
    total = len(dataset)

    for i, item in enumerate(dataset, start=1):
        question  = item["question"]
        reference = item["reference"]
        filters   = item.get("filters", {})

        log.info("[%d/%d] %s", i, total, question[:70])

        # Step 1: retrieve
        results = retriever.retrieve(question, top_k=top_k, top_n=top_n, **filters)

        # Step 2: generate
        resp = generator.generate(question, results)

        log.info("  answer: %s", resp.answer[:80])

        samples.append(SingleTurnSample(
            user_input=question,
            retrieved_contexts=resp.contexts,
            response=resp.answer,
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
    print("GENERATOR EVALUATION SUMMARY")
    print("=" * 60)
    for col in metric_cols:
        mean  = df[col].mean()
        worst = df[col].min()
        best  = df[col].max()
        print(f"  {col:<28}  mean={mean:.4f}  min={worst:.4f}  max={best:.4f}")
    print("─" * 60)
    print(f"  {'Samples evaluated':<28}  {n_samples}")
    print("=" * 60)

    # Diagnostic hint based on score pattern
    scores = {col: df[col].mean() for col in metric_cols}
    faith = scores.get("faithfulness", None)
    fact  = scores.get("factual_correctness", None)
    if faith is not None and fact is not None:
        print()
        if faith >= 0.8 and fact >= 0.8:
            print("  ✓ Pipeline looks healthy.")
        elif faith < 0.6 and fact >= 0.7:
            print("  ⚠ Low faithfulness: generator may be using parametric knowledge.")
            print("    Check the system prompt in generator.py.")
        elif faith >= 0.7 and fact < 0.6:
            print("  ⚠ Low factual correctness: retriever may be returning wrong chunks.")
            print("    Run eval_retriever.py to investigate retrieval quality.")
        else:
            print("  ✗ Both scores low: review retriever and generator.")
    print()


def save_results(result, output_path: Path) -> None:
    """Disabled: results now output only to terminal."""
    pass


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate full RAG pipeline quality with Ragas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                   help="Path to the golden generator dataset JSON.")
    p.add_argument("--top-n", type=int, default=5,
                   help="Number of chunks to retrieve per question.")
    p.add_argument("--top-k", type=int, default=40,
                   help="Candidate pool size per retriever before fusion.")
    p.add_argument("--eval-model", default=EVAL_LLM,
                   help="OpenAI model used as the Ragas evaluator LLM.")
    p.add_argument("--gen-model", default=EVAL_LLM,
                   help="OpenAI model used by the generator to produce answers.")
    p.add_argument("--golden-only", action="store_true",
                   help="Dry-run: skip retrieval and generation, no DB needed.")
    p.add_argument("--database-url",   default=os.environ.get("DATABASE_URL", ""))
    p.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.openai_api_key:
        log.error("OPENAI_API_KEY is not set.")
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
        ChatOpenAI(model=args.eval_model, api_key=args.openai_api_key, temperature=0)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=EMBED_MODEL, api_key=args.openai_api_key)
    )

    if args.golden_only:
        log.info("--golden-only: dry-run, no retrieval or generation.")
        samples = [
            SingleTurnSample(
                user_input=item["question"],
                retrieved_contexts=["(dry-run — no retrieval performed)"],
                response="(dry-run — no answer generated)",
                reference=item["reference"],
            )
            for item in dataset
        ]
    else:
        generator = Generator.from_credentials(
            args.openai_api_key, model=args.gen_model
        )
        with Retriever.from_credentials(args.database_url) as r:
            samples = build_samples(dataset, r, generator, args.top_k, args.top_n)

    if not samples:
        log.error("No samples produced.")
        return 1

    log.info("Evaluating %d sample(s) …", len(samples))

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(),         # is the answer grounded in the chunks?
            FactualCorrectness(),   # does the answer match the reference?
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    print_summary(result, len(samples))
    return 0


if __name__ == "__main__":
    sys.exit(main())