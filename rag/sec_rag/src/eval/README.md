# Evaluation Suite for SEC RAG Pipeline

This module contains scripts to evaluate the quality of the RAG pipeline at different stages using the Ragas evaluation framework.

## Dataset Strategy

### `golden_dataset.json` — Unified Golden Dataset

**Generation approach:**
- Extracted from actual chunks in `data/chunks/chunks.jsonl`
- 50+ factual question-reference pairs across 6 companies
- References are **guaranteed to exist** in the database
- Diverse coverage: revenue drivers, risks, segments, competitive landscape

**Why generate from chunks?**
- ✓ Questions and references are grounded in actual filings
- ✓ Enables validation that references can be retrieved
- ✓ Avoids human bias (uses document structure)
- ✓ Scalable: can generate more items by rerunning script

**Generation flow:**
```
chunks.jsonl (4,747 items)
    ↓
[Filter by length >= 200 chars]
    ↓
[Extract key sentences with facts/figures]
    ↓
[Generate questions from patterns]
    ↓
[Validate references exist in chunks]
    ↓
[Diversify across companies & periods]
    ↓
golden_dataset.json (50 items)
```

### Validating Dataset Quality

Before using the golden dataset:

```bash
# Generate fresh dataset from current chunks
uv run python src/eval/generate_golden_dataset.py --target-items 50

# Check validation stats
# Expected: 45+/50 valid items (90%+ validation rate)
```

## Evaluation Scripts

### 1. `eval_retriever.py` — Retriever Quality

**Scope:** Evaluates retrieval in isolation (no generator)

**Metrics:**
- **LLMContextRecall**: Coverage — did retriever find passages with the answer?
  - Range: 0-1 (higher is better)
  - Formula: `supported_sentences / total_sentences_in_reference`

- **LLMContextPrecision**: Relevance — are retrieved passages relevant or noise?
  - Range: 0-1 (higher is better)
  - Formula: `relevant_chunks / total_retrieved_chunks`

**What high/low scores mean:**
| Recall | Precision | Meaning |
|--------|-----------|---------|
| High | High | Excellent — finding right passages, no noise |
| High | Low | Noisy retrieval — finding answer but with junk |
| Low | High | Missing answer — precision good but incomplete |
| Low | Low | Both components need work |

**Usage:**
```bash
# Evaluate retriever with default dataset
uv run python src/eval/eval_retriever.py

# Evaluate with more chunks per question
uv run python src/eval/eval_retriever.py --top-n 10

# Dry-run without database
uv run python src/eval/eval_retriever.py --golden-only

# Custom dataset
uv run python src/eval/eval_retriever.py --dataset custom_dataset.json
```

**Output:**
```
RETRIEVER EVALUATION SUMMARY
============================================================
  llm_context_precision           mean=0.9500  min=0.8000  max=1.0000
  llm_context_recall              mean=0.8200  min=0.5000  max=1.0000
────────────────────────────────────────────────────────────
  Samples evaluated                50
============================================================
```

### 2. `eval_generator.py` — Full Pipeline Quality

**Scope:** Evaluates retriever + generator end-to-end

**Metrics:**
- **Faithfulness**: Is answer grounded in retrieved chunks?
  - Detects hallucination (LLM claims things not in context)
  - Formula: `grounded_claims / total_claims_in_response`

- **FactualCorrectness**: Does answer match the reference?
  - End-to-end quality — wrong retrieval OR generation both lower score
  - Formula: `correct_facts / total_facts`

**Score interpretation:**
```
High Faithfulness + High FactualCorrectness
  → Pipeline is working well

Low Faithfulness + High FactualCorrectness
  → Generator using parametric knowledge (luck, not grounding)

High Faithfulness + Low FactualCorrectness
  → Retriever returning wrong chunks; generator using them faithfully

Low Faithfulness + Low FactualCorrectness
  → Both components need work
```

**Usage:**
```bash
# Evaluate full pipeline
uv run python src/eval/eval_generator.py

# With custom generator model
uv run python src/eval/eval_generator.py --gen-model gpt-4o

# Dry-run without database
uv run python src/eval/eval_generator.py --golden-only
```

**Output:**
```
GENERATOR EVALUATION SUMMARY
============================================================
  faithfulness                    mean=0.8900  min=0.6000  max=1.0000
  factual_correctness             mean=0.7500  min=0.4000  max=1.0000
────────────────────────────────────────────────────────────
  Samples evaluated                50
============================================================

  ⚠ Low faithfulness: generator may be using parametric knowledge.
    Check the system prompt in generator.py.
```

### 3. `generate_golden_dataset.py` — Dataset Generation & Validation

**Purpose:** Generate golden dataset from actual chunks with validation

**What it does:**
1. Loads 4,747 chunks and 17 documents
2. Filters chunks by minimum length (200+ chars)
3. Extracts key sentences with facts, figures, comparisons
4. Generates questions using patterns (revenue, risks, segments, etc.)
5. Validates references exist in chunks (70%+ word overlap)
6. Diversifies across companies and fiscal periods
7. Produces 50 items by default

**Validation statistics:**
- **Valid**: Reference found in matching chunks
- **Partial**: Most of reference found (70%+ overlap)
- **Missing**: Reference not found (should be rare)

**Usage:**
```bash
# Generate 50-item dataset
uv run python src/eval/generate_golden_dataset.py

# Generate 100-item dataset
uv run python src/eval/generate_golden_dataset.py --target-items 100

# With stricter quality threshold
uv run python src/eval/generate_golden_dataset.py --min-chunk-length 300

# Custom output path
uv run python src/eval/generate_golden_dataset.py --output my_dataset.json
```

**Output:**
```
Loaded 4747 chunks.
Loaded 17 documents.
Filtered to 3210 chunks with length >= 200 chars.
Generated 892 candidate Q&A pairs.
Selected 50 items for final dataset.
Validating dataset...
Validation: 47 valid, 2 partial, 1 missing out of 50 total

✓ Golden dataset generated: src/eval/golden_dataset.json
  Items: 50
  Valid: 47/50
```

## Ragas Library Concepts

### Key Terms

| Term | Meaning |
|------|---------|
| **Golden Dataset** | Ground-truth Q&A pairs with reference answers |
| **SingleTurnSample** | One evaluation sample: {question, contexts, reference, response} |
| **EvaluationDataset** | Collection of samples |
| **Metric** | LLM-based scorer (e.g., LLMContextRecall) |
| **evaluate()** | Runs metrics across dataset, returns scores |

### Evaluation Hierarchy

```
Level 1: Retriever Eval (eval_retriever.py)
  Isolates: Does retriever find relevant chunks?
  Metrics: LLMContextRecall, LLMContextPrecision

Level 2: Generator Eval (eval_generator.py)
  Isolates: Does generator produce grounded answers?
  Metrics: Faithfulness, FactualCorrectness

Level 3: End-to-End Eval (future: eval_rag.py)
  Isolates: Does full pipeline answer questions correctly?
  Metrics: All of the above + custom metrics
```

## Workflow

**First time setup:**
```bash
# 1. Generate golden dataset from chunks
uv run python src/eval/generate_golden_dataset.py

# 2. Load/populate database
psql $DATABASE_URL -f db/schema.sql
uv run python src/ingest/embed_loader.py

# 3. Evaluate retriever in isolation
uv run python src/eval/eval_retriever.py
# Expected: recall >= 0.80, precision >= 0.85

# 4. Evaluate generator if retriever is good
uv run python src/eval/eval_generator.py
# Expected: faithfulness >= 0.85, factual_correctness >= 0.75
```

**Iterative improvement:**
```
Low Recall? → Check hybrid search (vector + FTS + RRF)
           → Increase top_k in retriever.py
           → Check if chunks are semantically similar

Low Precision? → Add filters to exclude irrelevant chunks
              → Adjust chunker settings (size, overlap)
              → Evaluate keyword search separately

Low Faithfulness? → Check system prompt in generator.py
                  → Reduce top_n (fewer chunks = less hallucination)
                  → Use stricter grounding instructions

Low Factual Correctness? → Improve retrieval quality first
                         → Then check generator model
                         → Consider caching for determinism
```

## Files

```
src/eval/
├── golden_dataset.json               # 50 Q&A pairs from chunks
├── eval_retriever.py                 # LLMContextRecall + Precision
├── eval_generator.py                 # Faithfulness + FactualCorrectness
├── generate_golden_dataset.py        # Script to generate & validate dataset
└── README.md                         # This file
```

## Tips

1. **Start with retriever eval** — validate retrieval before testing generation
2. **Use --golden-only** to test scripts without a database
3. **Monitor validation stats** — aim for 90%+ valid items
4. **Re-generate dataset** periodically as chunks grow
5. **Track scores over time** — CSV output enables trend analysis

## Further Reading

- [Ragas Documentation](https://docs.ragas.io)
- [LLM-as-Evaluator Pattern](https://arxiv.org/abs/2311.06351)
- [Grounded Generation for RAG](https://arxiv.org/abs/2309.09248)
