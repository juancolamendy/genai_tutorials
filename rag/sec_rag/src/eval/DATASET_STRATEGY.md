# Golden Dataset Strategy

## Overview

We use a **data-driven approach** to build the golden dataset by extracting real Q&A pairs from the actual chunks in `data/chunks/chunks.jsonl`.

This approach ensures:
- ✅ Questions and references are grounded in real SEC filings
- ✅ All references **can actually be retrieved** from the database
- ✅ No human bias in question formulation
- ✅ Scalable: generate more items by rerunning the script
- ✅ Reproducible: same chunks → same questions

## How It Works

### 1. **Data Source**

We have 4,747 chunks across 17 documents:

```
Companies:  AAPL (3), MSFT (3), NVDA (3), META (2), GOOGL (3), AMZN (3)
Time range: FY2023 → FY2026 (varies by company)
Total size: ~50 MB of SEC 10-K content
```

### 2. **Question Generation Pipeline**

```
Raw chunk (1.5 KB)
    ↓
[Extract key sentences with facts/figures]
    ↓
[Apply question templates]
    ↓
Candidate questions (10s per chunk)
    ↓
[Diversify & balance]
    ↓
Final questions (50 items)
```

### 3. **Question Patterns**

We generate questions using patterns found in real SEC filings:

**Pattern 1: Risk Disclosure**
```
Chunk: "...concentration of manufacturing in China as a key 
        supply chain risk, including exposure to geopolitical 
        tensions..."

Question: "What risks does Apple disclose in its 10-K?"
Reference: "Apple discloses concentration of manufacturing in China 
           as a key supply chain risk, including exposure to 
           geopolitical tensions..."
```

**Pattern 2: Revenue Sources**
```
Chunk: "...iPhone, which accounts for the largest share of its 
       net sales, supplemented by services, Mac, iPad, and 
       wearables."

Question: "What is Apple's primary source of revenue?"
Reference: "The iPhone accounts for the largest share of Apple's 
           net sales, supplemented by services, Mac, iPad, and 
           wearables."
```

**Pattern 3: Business Segments**
```
Chunk: "...The Americas segment had the highest revenue among 
       Apple's geographic segments in fiscal 2023, with net 
       sales of approximately $162.6 billion."

Question: "What are Apple's key geographic segments?"
Reference: "The Americas segment had the highest revenue among 
           Apple's geographic segments in fiscal 2023, with net 
           sales of approximately $162.6 billion."
```

### 4. **Validation**

Each reference is validated to ensure it's actually retrievable:

```python
# Validation algorithm:
1. Find chunks matching filters (ticker, form_type, fiscal_period)
2. Extract key words from reference
3. Check if 70%+ of reference words exist in matching chunks
4. Mark as VALID if found, PARTIAL if incomplete, MISSING if not found
```

**Expected validation rates:**
- VALID: 90%+ (reference found in chunks)
- PARTIAL: 5-7% (most of reference found)
- MISSING: <3% (shouldn't happen)

### 5. **Diversity Strategy**

To ensure broad coverage:

```
Distribute across:
├── Companies: ~8 items per company
├── Time periods: Mix of FY2023, FY2024, FY2025
├── Topics: Revenue, risks, segments, competitive landscape
└── Formats: Financial facts, disclosures, comparisons
```

## Why Not Manual Dataset?

| Approach | Pros | Cons |
|----------|------|------|
| **Data-driven** (ours) | Grounded in reality, scalable, reproducible | Generated questions may be generic |
| **Manual curation** | High-quality, specific questions | Expensive, biased, not scalable |
| **Hybrid** | Best of both | Time-consuming |

We chose **data-driven** because:
1. Focuses on fundamental quality: "Can retriever find real facts?"
2. Scalable: Easy to generate 100+ items
3. Reproducible: Same script + same chunks = same dataset
4. Transparent: Anyone can verify that questions come from chunks

## Extending the Dataset

### Add more items

```bash
uv run python src/eval/generate_golden_dataset.py --target-items 100
```

### Change quality threshold

```bash
# Stricter: only use longer chunks with more substance
uv run python src/eval/generate_golden_dataset.py --min-chunk-length 300
```

### Update with new filings

```bash
# After adding new chunks to data/chunks/chunks.jsonl
uv run python src/eval/generate_golden_dataset.py --target-items 100
```

## Evaluation Metrics vs. This Dataset

### What This Dataset Enables

**LLMContextRecall**
- Measures: Can retriever find sentences from the reference answer?
- Enabled by: Reference answers that are factual excerpts from chunks
- Example: Reference = "Apple discloses concentration in China" → Retriever must find this

**LLMContextPrecision**
- Measures: Are retrieved chunks relevant to the question?
- Enabled by: Questions that have clear right/wrong answers
- Example: "What risks does Apple disclose?" → Irrelevant chunks get low scores

**Faithfulness** (for eval_generator.py)
- Measures: Does generated answer use only the chunks?
- Enabled by: Questions where the answer is in the chunks
- Example: If chunks don't mention "Japan", good answer won't say it

## Future Improvements

1. **Add adversarial questions**
   - "What does Apple NOT disclose about..."
   - "How did X change from 2022 to 2023?"
   - Questions that require synthesis across multiple chunks

2. **Add unanswerable questions**
   - Questions where the answer is not in any chunk
   - Test that retriever knows when to say "not found"
   - Example: "What is Apple's revenue projection for 2030?"

3. **Multi-hop questions**
   - Questions requiring facts from 2+ chunks
   - Example: "What is Microsoft's largest business segment by operating income?"
   - Requires: AWS revenue (Chunk A) + profit margin (Chunk B)

4. **Temporal reasoning**
   - Cross-year comparisons
   - Example: "How did AWS revenue growth compare between FY2023 and FY2024?"

## Running the Full Evaluation

Once dataset is generated:

```bash
# 1. Set up environment
export DATABASE_URL=postgresql://...
export OPENAI_API_KEY=sk-...

# 2. Create schema
psql $DATABASE_URL -f db/schema.sql

# 3. Load embeddings
uv run python src/ingest/embed_loader.py

# 4. Evaluate retriever
uv run python src/eval/eval_retriever.py
# Output: retriever_20240625_143022.csv

# 5. Evaluate generator (if retriever looks good)
uv run python src/eval/eval_generator.py
# Output: generator_20240625_143045.csv
```

## Questions?

- See `generate_golden_dataset.py` for extraction logic
- See `eval_retriever.py` for retrieval evaluation
- See `eval_generator.py` for end-to-end evaluation
- See `README.md` for detailed documentation
