# SEC RAG Pipeline

A retrieval-augmented generation (RAG) system for querying SEC 10-K filings from
Apple, Microsoft, Nvidia, Meta, Google (Alphabet), and Amazon.

```
fetch_sec.py          download HTML filings from EDGAR
    ↓
converter.py          HTML → Markdown  (Docling)
    ↓
chunker.py            Markdown → documents.jsonl + chunks.jsonl  (HybridChunker)
    ↓
embed_load.py         embed chunks + load into Postgres/pgvector
    ↓
retriever.py          hybrid search (vector + full-text + RRF)
    ↓
eval_rag.py           evaluate with Ragas
```

---

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | ≥ 3.11 | [python.org](https://www.python.org) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Docker Desktop | latest | [docker.com](https://www.docker.com/products/docker-desktop/) |

---

## Project layout

```
sec-rag/
├── README.md
├── schema.sql               Postgres DDL (tables, indexes, HNSW)
├── fetch_sec.py             Step 1 — download filings
├── converter.py             Step 2 — HTML → Markdown
├── chunker.py               Step 3 — chunk Markdown
├── embed_load.py            Step 4 — embed + load into Postgres
├── fusion.py                RRF helper (used by retriever)
├── queries.py               SQL builders (used by retriever)
├── retriever.py             Hybrid retriever (CLI + library)
├── eval_rag.py              Ragas evaluation
├── start_postgres.sh        Docker helper for local Postgres
└── data/
    ├── html/                downloaded HTML filings
    ├── markdown/            converted Markdown
    ├── documents.jsonl      filing metadata (output of chunker)
    └── chunks.jsonl         chunk records ready for embedding
```

---

## 1. Clone and create the project

## 2. Install dependencies

```bash
# initialise a uv project (creates .venv automatically)
uv init .

# core pipeline
uv add httpx docling "docling-core[chunking-openai]" tiktoken openai "psycopg[binary]" pgvector

# evaluation
uv add ragas langchain-openai langchain-community
```

> **Note:** `docling` pulls PyTorch on first install (~1 GB). The HTML→Markdown
> pipeline uses only the lightweight HTML backend (no GPU, no model downloads).

---

## 3. Set environment variables

```bash
export OPENAI_API_KEY="sk-..."
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/sec_rag"
```

Add both to your `~/.zshrc` to avoid re-exporting on every session.

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

## 6. Run the pipeline

### Step 1 — Fetch filings from EDGAR

```bash
uv run python fetch_sec.py
```

Downloads the most recent 10-K for each company into `data/html/`:

```
data/html/
  AAPL/aapl-fy2024.htm
  AAPL/aapl-fy2024.meta.json
  MSFT/msft-fy2024.htm
  ...
```

Options:

```bash
uv run python fetch_sec.py --filings 3      # last 3 years per company
uv run python fetch_sec.py --force          # re-download existing files
```

---

### Step 2 — Convert HTML → Markdown

```bash
uv run python converter.py
```

Reads `data/html/`, writes `data/markdown/`. Docling preserves financial
table structure as Markdown pipes. Skips already-converted files.

```bash
uv run python converter.py --force          # re-convert everything
```

---

### Step 3 — Chunk Markdown

```bash
uv run python chunker.py
```

Reads `data/markdown/`, writes two flat files:

- `data/documents.jsonl` — one record per filing
- `data/chunks.jsonl` — one record per chunk, `content` field ready for the
  embedding API

```bash
uv run python chunker.py --max-tokens 800   # default, change if needed
uv run python chunker.py --force
```

---

### Step 4 — Embed and load into Postgres

```bash
uv run python embed_load.py
```

For each chunk: calls `text-embedding-3-large`, wraps the result in `halfvec`,
and upserts into the `chunks` table. Documents are upserted first. The script
is idempotent — re-running adds new filings without duplicating existing ones.

```bash
uv run python embed_load.py --batch-size 64     # smaller batches if rate-limited
```

---

## 7. Query

### As a CLI

```bash
uv run python retriever.py "What are Apple's main supply chain risks?"

# with filters
uv run python retriever.py "How did revenue change in 2023?" \
    --ticker AAPL --form-type 10-K --fiscal-period FY2023 --top-n 5
```

### As a library

```python
from retriever import Retriever

with Retriever.from_env() as r:
    results = r.retrieve(
        "What risks does Microsoft disclose for its cloud business?",
        ticker="MSFT",
        form_type="10-K",
        top_n=10,
    )
    for res in results:
        print(res.rrf_score, res.payload["section"])
        print(res.payload["content"])
```

---

## 8. Evaluate

```bash
uv run python eval_rag.py
```

Runs 10 hand-crafted SEC questions through the live retriever, generates
answers with `gpt-4o`, and scores with three Ragas metrics:

| Metric | What it measures |
|---|---|
| `LLMContextRecall` | Did the retriever surface the right chunks? |
| `Faithfulness` | Is the answer grounded in the retrieved context? |
| `FactualCorrectness` | Does the answer match the reference answer? |

Results are printed to the console and saved to `data/eval_YYYYMMDD_HHMMSS.csv`.

Options:

```bash
# synthesise additional Q&A from your Markdown corpus
uv run python eval_rag.py --generate --testset-size 10

# dry-run — test Ragas integration without a running database
uv run python eval_rag.py --golden-only
```

---

## Module reference

| File | Role |
|---|---|
| `schema.sql` | DDL — tables, B-tree indexes, HNSW index, GIN full-text index |
| `fetch_sec.py` | Downloads 10-K HTML + sidecar `.meta.json` from EDGAR |
| `converter.py` | Converts HTML to Markdown using Docling |
| `chunker.py` | Section-aware chunking with Docling `HybridChunker` + tiktoken |
| `embed_load.py` | Calls OpenAI embeddings API and upserts into Postgres |
| `fusion.py` | Pure-Python Reciprocal Rank Fusion — no external dependencies |
| `queries.py` | SQL builders for vector search and full-text search |
| `retriever.py` | Orchestrates embedding → vector search → FTS → RRF |
| `eval_rag.py` | Ragas evaluation with golden dataset and optional synthesis |
| `start_postgres.sh` | Docker helper — start, stop, reset, psql, logs |

---

## Troubleshooting

**`psql: could not connect to server`**
Postgres is not running. Run `start_postgres.sh` and check `start_postgres.sh status`.

**`ERROR: type "halfvec" does not exist`**
The schema was applied before the `vector` extension was created, or pgvector < 0.7.0
is installed. Run `start_postgres.sh reset` to recreate the container with the current
image, then reapply the schema.

**`openai.RateLimitError` during embed_load**
Reduce the batch size: `uv run python embed_load.py --batch-size 32`.

**`HTTP 403` from EDGAR during fetch_sec**
The SEC requires a valid `User-Agent` header. Edit the `HEADERS` dict in
`fetch_sec.py` and replace the email with your own.

**Docling takes a long time on first run**
The HTML backend is fast (no models). If you see model downloads, you may have
accidentally triggered the PDF pipeline — confirm you are pointing at `.htm` files.
