"""
LLM-as-Judge — Agno
===================
Evaluates agent outputs across three dimensions:
  - Faithfulness  : response grounded in retrieved context
  - Relevance     : response addresses the user query
  - Task Success  : agent accomplished the user goal

Setup
-----
    export ANTHROPIC_API_KEY="your-anthropic-api-key"

Run
---
    uv run main_agno.py
    uv run main_agno.py --verbose             # show per-case reasoning
    uv run main_agno.py --threshold 0.9       # stricter pass bar
    uv run main_agno.py --model claude-opus-4-20250514  # higher quality judge
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

# ── Agno ──────────────────────────────────────────────────────────────────────
from agno.agent import Agent
from agno.models.anthropic import Claude

# ── OpenTelemetry (console exporter — zero infra needed for local dev) ────────
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# ─────────────────────────────────────────────────────────────────────────────
# CLI args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-as-Judge for Agno agents")
    p.add_argument("--model",     default="claude-haiku-4-5-20251001",
                   help="Claude model to use as judge (default: claude-haiku-4-5-20251001)")
    p.add_argument("--threshold", type=float, default=0.7,
                   help="Per-case pass threshold 0–1 (default: 0.7)")
    p.add_argument("--ci-threshold", type=float, default=0.70,
                   help="CI batch pass-rate threshold 0–1 (default: 0.70)")
    p.add_argument("--verbose",   action="store_true",
                   help="Print per-dimension reasoning for every case")
    p.add_argument("--otel",      action="store_true",
                   help="Enable OTel console span output (noisy — off by default)")
    return p.parse_args()

ARGS = parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("llm_judge")

# ─────────────────────────────────────────────────────────────────────────────
# OpenTelemetry — console exporter (swap for OTLP in production)
# ─────────────────────────────────────────────────────────────────────────────

_provider = TracerProvider()
if ARGS.otel:
    _provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(_provider)
tracer = trace.get_tracer("llm_judge_agno")


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

class TaskOutcome(str, Enum):
    SUCCESS   = "success"
    PARTIAL   = "partial"
    FAILED    = "failed"
    ESCALATED = "escalated"


@dataclass
class DimensionScore:
    """Score + reasoning for one evaluation dimension."""
    dimension : str
    score     : float   # 0.0 – 1.0
    reason    : str
    latency_ms: float = 0.0


@dataclass
class JudgeResult:
    """Aggregated result of one evaluation run."""
    run_id       : str
    query        : str
    response     : str
    framework    : str
    timestamp    : str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    faithfulness : Optional[DimensionScore] = None
    relevance    : Optional[DimensionScore] = None
    task_success : Optional[DimensionScore] = None
    overall_score: float = 0.0
    passed       : bool  = False
    threshold    : float = 0.7

    def compute_overall(self) -> None:
        scores = [
            d.score
            for d in [self.faithfulness, self.relevance, self.task_success]
            if d is not None
        ]
        self.overall_score = sum(scores) / len(scores) if scores else 0.0
        self.passed        = self.overall_score >= self.threshold

    def as_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Judge agent
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """
You are an impartial AI evaluation judge.
Your sole job is to score an AI agent's response on the requested dimension.

Rules:
- Be objective and consistent.
- Base your score ONLY on what is provided — never assume missing information.
- Return ONLY a valid JSON object on a single line.
- JSON schema: {"score": <0 or 1>, "reason": "<one sentence>"}
- score 1 = criterion met, score 0 = criterion not met.
- No markdown, no preamble, no explanation outside the JSON.
"""


def build_judge_agent(model: str) -> Agent:
    """Create a stateless Agno Agent used as the evaluation judge."""
    return Agent(
        name        = "llm-judge",
        model       = Claude(id=model),
        description = "Impartial evaluator that scores agent responses.",
        instructions= [JUDGE_SYSTEM_PROMPT],
        # Keep the judge stateless and lightweight — no history, no storage
        add_history_to_context = False,
        markdown    = False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

def faithfulness_prompt(context: str, response: str) -> str:
    return f"""
Evaluate FAITHFULNESS.

Context provided to the agent:
\"\"\"
{context}
\"\"\"

Agent response:
\"\"\"
{response}
\"\"\"

Criterion: Does the response contain ONLY information present in the context?
Score 1 if every claim is supported by the context.
Score 0 if the response adds facts not found in the context.

Return JSON: {{"score": 0 or 1, "reason": "one sentence"}}
""".strip()


def relevance_prompt(query: str, response: str) -> str:
    return f"""
Evaluate RELEVANCE.

User query:
\"\"\"
{query}
\"\"\"

Agent response:
\"\"\"
{response}
\"\"\"

Criterion: Does the response directly and completely address the user query?
Score 1 if fully relevant.
Score 0 if off-topic, incomplete, or evasive.

Return JSON: {{"score": 0 or 1, "reason": "one sentence"}}
""".strip()


def task_success_prompt(query: str, response: str) -> str:
    return f"""
Evaluate TASK SUCCESS.

User goal:
\"\"\"
{query}
\"\"\"

Agent response:
\"\"\"
{response}
\"\"\"

Criterion: Did the agent successfully help the user accomplish their goal?
Score 1 if the goal is achieved or clear actionable next steps are given.
Score 0 if the goal is not met, the response is vague, or an error occurred.

Return JSON: {{"score": 0 or 1, "reason": "one sentence"}}
""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# Core runner helper
# ─────────────────────────────────────────────────────────────────────────────

async def _invoke_judge(
    agent    : Agent,
    prompt   : str,
    dimension: str,
) -> DimensionScore:
    """
    Send one evaluation prompt to the judge agent and return a DimensionScore.
    Returns score=0.0 with an error reason if the call or parse fails.

    Note: unlike ADK's streaming-event runner, Agno's `arun()` returns a
    single RunOutput object with `.content` already populated — no event
    loop needed to accumulate text.
    """
    t0 = time.time()

    with tracer.start_as_current_span(f"judge.{dimension}") as span:
        span.set_attribute("judge.dimension",  dimension)
        span.set_attribute("judge.prompt_len", len(prompt))

        try:
            # Each call gets a fresh session_id so judge calls never share
            # conversation history (mirrors ADK's per-call session_id)
            result = await agent.arun(
                prompt,
                session_id=str(uuid.uuid4()),
            )
            raw_text = result.content or ""

            # Agno exposes token usage on result.metrics as a Metrics object
            # (attribute access, not a dict) — surface it on the span the
            # same way the ADK version does with usage_metadata
            if result.metrics:
                span.set_attribute(
                    "llm.prompt_tokens",
                    result.metrics.input_tokens or 0
                )
                span.set_attribute(
                    "llm.completion_tokens",
                    result.metrics.output_tokens or 0
                )

        except Exception as exc:
            log.error("Judge invocation failed for %s: %s", dimension, exc)
            span.record_exception(exc)
            return DimensionScore(
                dimension  = dimension,
                score      = 0.0,
                reason     = f"Invocation error: {exc}",
                latency_ms = (time.time() - t0) * 1000,
            )

        latency_ms = (time.time() - t0) * 1000
        span.set_attribute("judge.latency_ms",   latency_ms)
        span.set_attribute("judge.raw_response", raw_text[:500])

        score, reason = _parse_judge_response(raw_text, dimension)
        span.set_attribute("judge.score",  score)
        span.set_attribute("judge.reason", reason)

        return DimensionScore(
            dimension  = dimension,
            score      = score,
            reason     = reason,
            latency_ms = latency_ms,
        )


def _parse_judge_response(raw: str, dimension: str) -> tuple[float, str]:
    """
    Parse {"score": 0|1, "reason": "..."} from judge output.
    Strips markdown fences; extracts first valid JSON object found.
    """
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\{.*?\}", clean, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response")
        data   = json.loads(match.group())
        score  = float(data.get("score", 0))
        reason = str(data.get("reason", "No reason provided"))
        score  = max(0.0, min(1.0, score))   # clamp to [0, 1]
        return score, reason
    except Exception as exc:
        log.warning(
            "Failed to parse judge response for %s: %s | raw=%r",
            dimension, exc, raw[:200],
        )
        return 0.0, f"Parse error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

class LLMJudge:
    """
    LLM-as-Judge evaluator for Agno agents.

    Evaluates three dimensions concurrently:
        faithfulness, relevance, task_success

    Usage
    -----
        judge  = LLMJudge()
        result = await judge.evaluate(
            query     = "Triage ticket TKT-101 and assign it to the backend team",
            response  = "Ticket TKT-101 has been labelled 'backend' and assigned. ID confirmed.",
            context   = "TKT-101: login service throws 500 on POST /auth. Priority: high.",
        )
        print(result.overall_score, result.passed)
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001", threshold: float = 0.7) -> None:
        self.threshold = threshold
        self._agent    = build_judge_agent(model)
        log.info("LLMJudge ready | model=%s  threshold=%.2f", model, threshold)

    # ── Single evaluation ─────────────────────────────────────────────────────

    async def evaluate(
        self,
        query    : str,
        response : str,
        context  : str = "",
        run_id   : str = "",
        framework: str = "agno",
    ) -> JudgeResult:
        """
        Evaluate one agent response across all three dimensions in parallel.

        Parameters
        ----------
        query     : the original user query / goal
        response  : the agent's response to evaluate
        context   : retrieved context passed to the agent (used for faithfulness)
        run_id    : identifier of the production run being evaluated
        framework : label stored on the result
        """
        run_id = run_id or str(uuid.uuid4())

        with tracer.start_as_current_span("llm_judge.evaluate") as span:
            span.set_attribute("judge.run_id",    run_id)
            span.set_attribute("judge.framework", framework)
            span.set_attribute("judge.query",     query[:200])

            faith_score, rel_score, task_score = await asyncio.gather(
                _invoke_judge(
                    self._agent,
                    faithfulness_prompt(context, response),
                    "faithfulness",
                ),
                _invoke_judge(
                    self._agent,
                    relevance_prompt(query, response),
                    "relevance",
                ),
                _invoke_judge(
                    self._agent,
                    task_success_prompt(query, response),
                    "task_success",
                ),
            )

            result = JudgeResult(
                run_id       = run_id,
                query        = query,
                response     = response,
                framework    = framework,
                faithfulness = faith_score,
                relevance    = rel_score,
                task_success = task_score,
                threshold    = self.threshold,
            )
            result.compute_overall()

            span.set_attribute("judge.overall_score", result.overall_score)
            span.set_attribute("judge.passed",        result.passed)

            self._log_result(result)
            return result

    # ── Batch evaluation ──────────────────────────────────────────────────────

    async def evaluate_batch(
        self,
        samples  : list[dict],
        framework: str = "agno",
    ) -> list[JudgeResult]:
        """
        Evaluate a list of samples concurrently.

        Each dict must contain: query, response
        Optional keys        : context, run_id
        """
        log.info("Batch evaluation: %d samples", len(samples))
        tasks = [
            self.evaluate(
                query     = s["query"],
                response  = s["response"],
                context   = s.get("context", ""),
                run_id    = s.get("run_id", str(uuid.uuid4())),
                framework = framework,
            )
            for s in samples
        ]
        return await asyncio.gather(*tasks)

    # ── Aggregate stats ───────────────────────────────────────────────────────

    @staticmethod
    def summarize_batch(results: list[JudgeResult]) -> dict:
        """Return aggregate statistics for a batch of JudgeResults."""
        if not results:
            return {}

        def avg(vals: list[float]) -> float:
            return round(sum(vals) / len(vals), 3) if vals else 0.0

        return {
            "total"            : len(results),
            "passed"           : sum(1 for r in results if r.passed),
            "pass_rate"        : avg([1.0 if r.passed else 0.0 for r in results]),
            "avg_overall"      : avg([r.overall_score for r in results]),
            "avg_faithfulness" : avg([r.faithfulness.score for r in results if r.faithfulness]),
            "avg_relevance"    : avg([r.relevance.score    for r in results if r.relevance]),
            "avg_task_success" : avg([r.task_success.score for r in results if r.task_success]),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _log_result(result: JudgeResult) -> None:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        log.info(
            "%s  run_id=%-22s  overall=%.2f  "
            "faith=%.1f  rel=%.1f  task=%.1f",
            status,
            result.run_id,
            result.overall_score,
            result.faithfulness.score  if result.faithfulness  else -1,
            result.relevance.score     if result.relevance      else -1,
            result.task_success.score  if result.task_success   else -1,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CI gate helper
# ─────────────────────────────────────────────────────────────────────────────

async def run_ci_evaluation(
    judge         : LLMJudge,
    golden_dataset: list[dict],
    min_pass_rate : float = 0.85,
    verbose       : bool  = False,
) -> bool:
    """
    Run the full golden dataset through the judge.
    Returns True (CI passes) if pass_rate >= min_pass_rate.

    Logs every failing case with per-dimension reasoning so you know
    exactly which cases to fix.
    """
    log.info(
        "CI evaluation started | samples=%d  required_pass_rate=%.0f%%",
        len(golden_dataset), min_pass_rate * 100,
    )

    results = await judge.evaluate_batch(golden_dataset)
    summary = judge.summarize_batch(results)
    passed  = summary["pass_rate"] >= min_pass_rate

    # ── Summary banner ────────────────────────────────────────────────────────
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  CI Result : {'PASS ✓' if passed else 'FAIL ✗'}")
    print(f"  Samples   : {summary['total']}  |  Passed: {summary['passed']}")
    print(f"  Pass rate : {summary['pass_rate']:.1%}  (threshold: {min_pass_rate:.0%})")
    print(f"  Avg overall     : {summary['avg_overall']:.3f}")
    print(f"  Avg faithfulness: {summary['avg_faithfulness']:.3f}")
    print(f"  Avg relevance   : {summary['avg_relevance']:.3f}")
    print(f"  Avg task success: {summary['avg_task_success']:.3f}")
    print(bar)

    # ── Per-case detail ───────────────────────────────────────────────────────
    print("\nPer-case breakdown:")
    for r in results:
        icon = "✓" if r.passed else "✗"
        print(
            f"  {icon}  [{r.overall_score:.2f}]  {r.query[:65]}"
        )
        if verbose or not r.passed:
            if r.faithfulness:
                fi = "✓" if r.faithfulness.score >= 0.5 else "✗"
                print(f"       faithfulness {fi} ({r.faithfulness.score:.0f})  — {r.faithfulness.reason}")
            if r.relevance:
                ri = "✓" if r.relevance.score >= 0.5 else "✗"
                print(f"       relevance    {ri} ({r.relevance.score:.0f})  — {r.relevance.reason}")
            if r.task_success:
                ti = "✓" if r.task_success.score >= 0.5 else "✗"
                print(f"       task_success {ti} ({r.task_success.score:.0f})  — {r.task_success.reason}")

    print()
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Golden dataset
# Four cases: 2 should pass, 2 are intentionally broken to test the judge.
# ─────────────────────────────────────────────────────────────────────────────

GOLDEN_DATASET: list[dict] = [
    # ── Case 1: clean success ─────────────────────────────────────────────────
    {
        "run_id"  : "case-001",
        "query"   : "Triage ticket TKT-101 and assign it to the backend team",
        "response": (
            "Ticket TKT-101 has been triaged as high priority and assigned "
            "to the backend team. Assignment ID: TKT-101-BE."
        ),
        "context" : (
            "TKT-101: login service throws HTTP 500 on POST /auth. "
            "Priority: high. Unassigned."
        ),
    },
    # ── Case 2: correct policy answer ─────────────────────────────────────────
    {
        "run_id"  : "case-002",
        "query"   : "What is the SLA for high-priority tickets?",
        "response": (
            "High-priority tickets must receive an initial response within "
            "2 hours and a resolution within 8 hours."
        ),
        "context" : (
            "SLA policy: P1 (high) — first response ≤2h, resolution ≤8h. "
            "P2 (medium) — first response ≤8h, resolution ≤24h."
        ),
    },
    # ── Case 3: faithfulness failure — agent adds info not in context ──────────
    {
        "run_id"  : "case-003",
        "query"   : "What components does TKT-202 affect?",
        "response": (
            "TKT-202 affects the authentication service and also the "
            "payment gateway and the notification microservice."  # ← hallucinated
        ),
        "context" : "TKT-202: authentication service returns 401 on valid tokens.",
    },
    # ── Case 4: relevance failure — agent goes completely off-topic ────────────
    {
        "run_id"  : "case-004",
        "query"   : "Find all open tickets assigned to Alice",
        "response": "The deployment pipeline is currently healthy and all services are green.",
        "context" : "No tickets assigned to Alice found in the current sprint.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 60)
    print("  LLM-as-Judge — Agno  |  Ticket Processing Agent")
    print("=" * 60)
    print(f"  Model    : {ARGS.model}")
    print(f"  Threshold: {ARGS.threshold}  |  CI threshold: {ARGS.ci_threshold}")
    print()

    judge = LLMJudge(model=ARGS.model, threshold=ARGS.threshold)

    # ── Single evaluation demo ────────────────────────────────────────────────
    print("── Single evaluation (case-001) ──")
    single = await judge.evaluate(**{k: v for k, v in GOLDEN_DATASET[0].items()})
    print(json.dumps(single.as_dict(), indent=2, default=str))

    # ── Full CI batch ─────────────────────────────────────────────────────────
    print("\n── CI batch evaluation ──")
    ci_passed = await run_ci_evaluation(
        judge          = judge,
        golden_dataset = GOLDEN_DATASET,
        min_pass_rate  = ARGS.ci_threshold,
        verbose        = ARGS.verbose,
    )

    sys.exit(0 if ci_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
