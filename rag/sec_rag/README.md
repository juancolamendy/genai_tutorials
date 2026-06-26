# SEC RAG Pipeline

A retrieval-augmented generation (RAG) system for querying SEC 10-K filings from
Apple, Microsoft, Nvidia, Meta, Google (Alphabet), and Amazon.

**Architecture:**
- **Embeddings**: Ollama (nomic-embed-text, 768-dim)
- **Retrieval**: Hybrid (vector search via pgvector + full-text search + RRF fusion)
- **Generation**: Claude 3.5 Haiku via Anthropic API
- **Evaluation**: Ragas (context recall, context precision, faithfulness, factual correctness)

```
sec_fetcher.py        download HTML filings from EDGAR
    ↓
converter.py          HTML → Markdown  (Docling)
    ↓
chunker.py            Markdown → documents + chunks (HybridChunker)
    ↓
embed_loader.py       embed chunks + load into Postgres/pgvector (Ollama)
    ↓
retriever.py          hybrid search (vector + full-text + RRF)
    ↓
generator.py          generate answers with Claude Haiku
    ↓
eval_retriever.py     evaluate retrieval quality (context recall/precision)
eval_generator.py     evaluate generation quality (faithfulness/factual correctness)
```

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | ≥ 3.11 | [python.org](https://www.python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker Desktop | latest | [docker.com](https://www.docker.com/products/docker-desktop/) |
| Ollama | latest | [ollama.ai](https://ollama.ai) — for embeddings (nomic-embed-text) |

**API Keys Required:**
- `ANTHROPIC_API_KEY` — Claude API key from [console.anthropic.com](https://console.anthropic.com)
- `DATABASE_URL` — PostgreSQL connection string

---

## Project Layout

```
sec-rag/
├── README.md
├── schema.sql                   Postgres DDL (tables, indexes, HNSW)
├── start_postgres.sh            Docker helper for local Postgres
│
├── src/
│   ├── tools/
│   │   └── sec_fetcher.py       Step 1 — download 10-K HTML from EDGAR
│   │
│   ├── ingest/
│   │   ├── converter.py         Step 2 — HTML → Markdown (Docling)
│   │   ├── chunker.py           Step 3 — Markdown → chunks (HybridChunker)
│   │   └── embed_loader.py      Step 4 — embed + load into Postgres (Ollama)
│   │
│   ├── embeddings/
│   │   └── embeddings.py        Embedding provider abstraction (Ollama, HuggingFace, OpenAI)
│   │
│   ├── retriever/
│   │   ├── retriever.py         Hybrid retriever (vector + FTS + RRF)
│   │   ├── queries.py           SQL builders (vector_search, keyword_search)
│   │   ├── fusion.py            Reciprocal Rank Fusion (RRF)
│   │   └── generator.py         Answer generation with Claude
│   │
│   └── eval/
│       ├── eval_retriever.py    Evaluate retrieval quality (context recall/precision)
│       ├── eval_generator.py    Evaluate generation quality (faithfulness/factual correctness)
│       ├── debug_retrieval.py   Debug a specific item (vector + keyword + RRF)
│       ├── analyze_retrieval_quality.py   Analyze failure patterns
│       ├── generate_golden_dataset_from_db.py   Generate golden dataset from DB chunks
│       └── golden_dataset.json  50 Q&A items for evaluation
│
└── data/
    ├── html/                    Downloaded HTML filings from EDGAR
    ├── markdown/                Converted Markdown
    ├── documents.jsonl          Filing metadata (1 per filing)
    └── chunks.jsonl             Chunk records ready for embedding
```

---

## 1. Clone and create the project

## 2. Install dependencies

```bash
# Initialize a uv project (creates .venv automatically)
uv init .

# Core pipeline (ingestion + retrieval)
uv add \
  httpx \
  docling "docling-core[chunking-openai]" \
  tiktoken \
  anthropic langchain-anthropic \
  "psycopg[binary]" pgvector \
  ollama

# Evaluation (Ragas + LLM)
uv add ragas langchain-community
```

**Optional:** If using HuggingFace embeddings:
```bash
uv add sentence-transformers
```

**Optional:** If using OpenAI embeddings:
```bash
uv add openai langchain-openai
```

> **Note:** `docling` pulls PyTorch on first install (~1 GB). The HTML→Markdown
> pipeline uses only the lightweight HTML backend (no GPU, no model downloads).

---

## 3. Set environment variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/sec_rag"
export EMBEDDING_PROVIDER="ollama"  # optional, defaults to Ollama
```

Add to your `~/.zshrc` (or `~/.bashrc`) to avoid re-exporting on every session.

**Optional:** If using HuggingFace or OpenAI embeddings instead of Ollama:
```bash
export EMBEDDING_PROVIDER="huggingface"  # or "openai"
export HUGGINGFACE_API_KEY="hf_..."      # if using HuggingFace
export OPENAI_API_KEY="sk-..."           # if using OpenAI
```

---

## 4. Start Postgres

---

## 5. Apply the schema

```bash
psql $DATABASE_URL -f schema.sql
```

```bash
psql 
\c sec_rag
```

This creates:
- `documents` table — one row per filing with metadata and B-tree indexes
- `chunks` table — one row per chunk with a `halfvec(3072)` embedding column,
  an HNSW index for cosine similarity, and a generated `tsvector` column for
  full-text search

Verify:

```bash
psql $DATABASE_URL -c "\dt"
psql $DATABASE_URL -c "\di"
```

---

## 6. Run the Ingestion Pipeline

### Step 1 — Fetch filings from EDGAR

```bash
uv run python src/tools/sec_fetcher.py
```

Downloads the most recent 10-K for each company into `data/html/`:

```
data/html/
  AAPL/aapl-fy2024.htm
  AAPL/aapl-fy2024.meta.json
  MSFT/msft-fy2024.htm
  ...
```

**Options:**
```bash
uv run python src/tools/sec_fetcher.py --filings 3      # last 3 years per company
uv run python src/tools/sec_fetcher.py --force          # re-download existing files
```

---

### Step 2 — Convert HTML → Markdown

```bash
uv run python src/ingest/converter.py
```

Reads `data/html/`, writes `data/markdown/`. Docling preserves financial
table structure as Markdown. Skips already-converted files.

**Options:**
```bash
uv run python src/ingest/converter.py --force          # re-convert everything
```

---

### Step 3 — Chunk Markdown

```bash
uv run python src/ingest/chunker.py
```

Reads `data/markdown/`, writes two JSONL files:

- `data/documents.jsonl` — one record per filing
- `data/chunks.jsonl` — one record per chunk, ready for embedding

**Options:**
```bash
uv run python src/ingest/chunker.py --max-tokens 800   # chunk size (default: 800)
uv run python src/ingest/chunker.py --force            # re-chunk everything
```

---

### Step 4 — Embed and Load into Postgres

```bash
uv run python src/ingest/embed_loader.py
```

For each chunk: embeds with Ollama (nomic-embed-text), wraps in `halfvec`,
and upserts into Postgres. Documents are upserted first. Idempotent —
re-running adds new filings without duplicating existing ones.

**Options:**
```bash
uv run python src/ingest/embed_loader.py --batch-size 64     # smaller if rate-limited
```

---

## 7. Retrieve

### As a CLI (Retriever only)

```bash
uv run python src/retriever/retriever.py "What are Apple's main supply chain risks?"

# with metadata filters
uv run python src/retriever/retriever.py "How did revenue change?" \
    --ticker AAPL --form-type 10-K --fiscal-period FY2023 --top-n 10
```

### As a CLI (Retriever + Generator)

```bash
uv run python src/retriever/generator.py "What are Apple's supply chain risks?" --ticker AAPL

# with custom models
uv run python src/retriever/generator.py "..." \
    --model claude-opus-4-8 --top-n 15 --top-k 150
```

### As a Library

```python
from src.retriever import Retriever
from src.retriever.generator import Generator

# Retrieval only
with Retriever.from_credentials(database_url) as r:
    results = r.retrieve(
        "What risks does Microsoft disclose for cloud?",
        ticker="MSFT",
        top_n=10,
    )
    for res in results:
        print(res.rrf_score, res.payload["content"][:100])

# Retrieval + Generation
gen = Generator.from_credentials(anthropic_api_key)
r = Retriever.from_credentials(database_url)
results = r.retrieve("...", top_n=10)
response = gen.generate("...", results)
print(response.answer)
```

---

## 8. Evaluate

### Retriever Evaluation (vector + FTS + RRF)

```bash
uv run python src/eval/eval_retriever.py --top-k 150 --top-n 20
```

Evaluates retrieval quality on 50 golden dataset items:

| Metric | Current | What it measures |
|---|---|---|
| **Context Recall** | 71.50% | % of reference sentences found in top-20 chunks |
| **Context Precision** | 10.59% | % of retrieved chunks relevant to question |

---

### Generator Evaluation (retriever + Claude)

```bash
uv run python src/eval/eval_generator.py --top-k 150 --top-n 20
```

Evaluates full pipeline (retrieval + generation) on 50 items:

| Metric | Current | What it measures |
|---|---|---|
| **Faithfulness** | 94.42% | % of answer claims supported by retrieved chunks |
| **Factual Correctness** | 9.34% | % of answer matching golden dataset reference |

**Note:** Low factual correctness is due to retriever finding only 71.5% of reference chunks.
Improving retrieval to 85%+ would likely improve factual correctness significantly.

---

### Debug a Specific Item

```bash
# inspect retrieval for item 0
uv run python src/eval/debug_retrieval.py --item 0 --top-k 150 --top-n 20

# or with a custom question
uv run python src/eval/debug_retrieval.py \
    --question "What are Apple's main risks?" \
    --ticker AAPL --top-k 150 --top-n 20
```

Shows vector search results, keyword search results, and final RRF-fused ranking.

---

### Analyze Retrieval Failures

```bash
uv run python src/eval/analyze_retrieval_quality.py --top-k 150 --top-n 20
```

Identifies which items fail and why (vector-only failures, keyword-only failures, etc.).

---

## Architecture Decisions

### Embeddings: Ollama (nomic-embed-text)
- **Why not OpenAI?** Reduced latency (local inference), no API costs, privacy
- **Why not HuggingFace?** Ollama is simpler to deploy (no GPU setup needed)
- **Trade-off:** Slightly lower semantic quality than OpenAI text-embedding-3-large, but still performs well (71.5% recall)
- **Vector size:** 768-dim (halfvec in PostgreSQL for 2x storage efficiency vs float32)

### Retrieval: Hybrid (Vector + Full-Text + RRF)
- **Vector search:** Fast semantic matching via pgvector HNSW index
- **Keyword search:** PostgreSQL full-text search (tsvector) for terminology-specific queries
- **Fusion:** Equal-weight Reciprocal Rank Fusion (RRF) combines both ranked lists
  - Tried weighting vector search higher (0.7/0.3), but recall dropped
  - Equal weights work best for SEC filings (terminology varies across companies)

### Generation: Claude Haiku via Anthropic API
- **Why Claude?** Superior reasoning for complex financial analysis vs other models
- **Why Haiku?** Fast, cost-effective, sufficient for Q&A synthesis tasks
- **Chain:** LangChain StructuredOutput parser for consistent JSON responses

### Evaluation: Ragas Metrics
- **Context Recall:** % of reference sentences supported by retrieved chunks (measures retriever completeness)
- **Context Precision:** % of retrieved chunks relevant to question (measures noise)
- **Faithfulness:** % of answer claims grounded in context (measures hallucination)
- **Factual Correctness (F1):** Token-level F1 with reference answer (measures answer quality)

---

## Performance Metrics

### Retrieval Quality (on 50-item golden dataset)

| Metric | Score | Interpretation |
|---|---|---|
| **Context Recall** | 71.50% | 71% of answer-supporting passages found in top-20 results |
| **Context Precision** | 10.59% | 11% of retrieved chunks are relevant; 89% are supporting context |

**How it works:**
- Vector search (Ollama embeddings) finds ~71% of supporting chunks
- Keyword search (PostgreSQL full-text) adds coverage for terminology-specific matches
- Reciprocal Rank Fusion combines both ranked lists

---

### Generation Quality (on 50-item golden dataset)

| Metric | Score | Interpretation |
|---|---|---|
| **Faithfulness** | 94.42% | 94% of generated claims are grounded in retrieved chunks |
| **Factual Correctness (F1)** | 9.34% | 9% of answers match golden dataset references exactly |

**Why low factual correctness?**
- Retriever finds supporting chunks 71.5% of the time (not 100%)
- When chunks are missing, generator produces partial/different answers
- F1 metric requires near-exact match with reference; paraphrases score lower
- High faithfulness (94%) indicates generator is grounded, not hallucinating

**Next steps to improve:**
- Push retriever recall to 80%+ (would improve factual correctness to ~40-50%)
- Use cross-encoder re-ranking to improve precision
- Expand queries with synonyms to catch terminology variance

---

## Module Reference

### Infrastructure
| File | Role |
|---|---|
| `schema.sql` | DDL — `documents` and `chunks` tables, HNSW index, GIN full-text index |
| `start_postgres.sh` | Docker helper — start, stop, reset Postgres, run psql |

### Ingestion Pipeline
| File | Role |
|---|---|
| `src/tools/sec_fetcher.py` | Downloads 10-K HTML + metadata from SEC EDGAR |
| `src/ingest/converter.py` | Converts HTML → Markdown (Docling) |
| `src/ingest/chunker.py` | Section-aware chunking with HybridChunker + tiktoken |
| `src/ingest/embed_loader.py` | Embeds chunks (Ollama) and upserts into Postgres |

### Retrieval & Generation
| File | Role |
|---|---|
| `src/embeddings/embeddings.py` | Embedding provider abstraction (Ollama, HuggingFace, OpenAI) |
| `src/retriever/queries.py` | SQL builders: `vector_search()`, `keyword_search()` |
| `src/retriever/fusion.py` | Reciprocal Rank Fusion (RRF) — pure Python, no dependencies |
| `src/retriever/retriever.py` | Orchestrates embedding → vector search → FTS → RRF |
| `src/retriever/generator.py` | Answer generation with Claude (LangChain chain) |

### Evaluation
| File | Role |
|---|---|
| `src/eval/eval_retriever.py` | Retrieval evaluation (Ragas context recall/precision) |
| `src/eval/eval_generator.py` | Full pipeline evaluation (faithfulness/factual correctness) |
| `src/eval/debug_retrieval.py` | Debug utility — inspect retrieval for a specific item |
| `src/eval/analyze_retrieval_quality.py` | Analyze failure patterns across dataset |
| `src/eval/generate_golden_dataset_from_db.py` | Generate golden dataset from actual DB chunks |
| `src/eval/golden_dataset.json` | 50 Q&A items for evaluation (generated from DB) |

---

## Troubleshooting

### Database
**`psql: could not connect to server`**
- Postgres is not running. Run `./start_postgres.sh up` and check `./start_postgres.sh status`.

**`ERROR: type "halfvec" does not exist`**
- The schema was applied before the `vector` extension was created, or pgvector < 0.7.0 is installed.
- Run `./start_postgres.sh reset` to recreate the container, then reapply `schema.sql`.

### Embeddings
**`ConnectionError: Failed to connect to Ollama at http://localhost:11434`**
- Ollama is not running. Start it: `ollama serve` (or use `Ollama.app` on macOS).
- Pull the embedding model: `ollama pull nomic-embed-text`

**`ModuleNotFoundError: No module named 'ollama'`**
- Install Ollama Python client: `uv add ollama`

### SEC Fetcher
**`HTTP 403` from EDGAR during fetch_sec**
- The SEC requires a valid `User-Agent` header. Edit `HEADERS` in `src/tools/sec_fetcher.py` and use your email.

### Docling
**Docling takes a long time on first run**
- The HTML backend is fast (no GPU, no models). If you see model downloads, you may have accidentally triggered the PDF pipeline.
- Verify you are pointing at `.htm` files, not PDFs.

### Generation
**`ModuleNotFoundError: No module named 'anthropic'`**
- Install Anthropic client: `uv add anthropic langchain-anthropic`

**`AuthenticationError: Invalid API key`**
- Verify `ANTHROPIC_API_KEY` is set: `echo $ANTHROPIC_API_KEY`

### Evaluation
**`ImportError: cannot import name 'LLMContextRecall'` (Ragas)**
- Ragas API changed between versions. Try: `uv add "ragas>=0.1.0"`
