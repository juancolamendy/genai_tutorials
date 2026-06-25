"""SQL queries for hybrid retrieval against the SEC RAG store.

Each function returns (sql_string, params_dict) ready to pass to
psycopg's ``cursor.execute(sql, params)``.

Keeping SQL here — away from retriever.py — makes it easy to tune, test,
and explain individual queries without touching orchestration logic.
"""

from __future__ import annotations

from pgvector import HalfVector


def vector_search(
    query_embedding: list[float],
    top_k: int = 40,
    ticker: str | None = None,
    form_type: str | None = None,
    fiscal_period: str | None = None,
) -> tuple[str, dict]:
    """Semantic search: nearest neighbours by cosine distance (halfvec <=>).

    Metadata filters (ticker, form_type, fiscal_period) are applied as WHERE
    clauses BEFORE the vector scan.  With HNSW this narrows the candidate rows
    the index traverses, but tight filters can hurt recall — see the note in
    schema.sql about hnsw.iterative_scan for pgvector >= 0.8.0.

    Returns top_k rows ordered by ascending cosine distance (closest first).
    """
    filters, params = _build_filters(ticker, form_type, fiscal_period)
    filters.append("c.content IS NOT NULL")  # Exclude malformed chunks
    where = ("WHERE " + " AND ".join(filters)) if filters else ""

    sql = f"""
        SELECT
            c.id          AS chunk_id,
            c.document_id,
            c.section,
            c.chunk_index,
            c.content,
            d.ticker,
            d.company,
            d.form_type,
            d.fiscal_period,
            d.filing_date,
            d.url,
            c.embedding <=> %(query_embedding)s  AS cos_dist
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        {where}
        ORDER BY c.embedding <=> %(query_embedding)s
        LIMIT %(top_k)s;
    """
    params["query_embedding"] = HalfVector(query_embedding)
    params["top_k"] = top_k
    return sql, params


def keyword_search(
    query_text: str,
    top_k: int = 40,
    ticker: str | None = None,
    form_type: str | None = None,
    fiscal_period: str | None = None,
) -> tuple[str, dict]:
    """Full-text search using Postgres ``websearch_to_tsquery``.

    ``websearch_to_tsquery`` accepts natural Google-style input:
    quoted phrases, OR, -, etc.  It gracefully ignores stop words and
    handles stemming, so "supply chains" matches "supply chain".

    Returns top_k rows ordered by descending ts_rank (best match first).
    """
    filters, params = _build_filters(ticker, form_type, fiscal_period)
    # tsv is a GENERATED ALWAYS tsvector column on chunks.content
    filters.append("c.tsv @@ websearch_to_tsquery('english', %(query_text)s)")
    filters.append("c.content IS NOT NULL")  # Exclude malformed chunks
    where = "WHERE " + " AND ".join(filters)

    sql = f"""
        SELECT
            c.id          AS chunk_id,
            c.document_id,
            c.section,
            c.chunk_index,
            c.content,
            d.ticker,
            d.company,
            d.form_type,
            d.fiscal_period,
            d.filing_date,
            d.url,
            ts_rank(c.tsv, websearch_to_tsquery('english', %(query_text)s)) AS fts_rank
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        {where}
        ORDER BY fts_rank DESC
        LIMIT %(top_k)s;
    """
    params["query_text"] = query_text
    params["top_k"] = top_k
    return sql, params


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _build_filters(
    ticker: str | None,
    form_type: str | None,
    fiscal_period: str | None,
) -> tuple[list[str], dict]:
    """Return (list_of_sql_conditions, params_dict) for optional metadata filters."""
    filters: list[str] = []
    params: dict = {}
    if ticker:
        filters.append("d.ticker = %(ticker)s")
        params["ticker"] = ticker.upper()
    if form_type:
        filters.append("d.form_type = %(form_type)s")
        params["form_type"] = form_type.upper()
    if fiscal_period:
        filters.append("d.fiscal_period = %(fiscal_period)s")
        params["fiscal_period"] = fiscal_period
    return filters, params