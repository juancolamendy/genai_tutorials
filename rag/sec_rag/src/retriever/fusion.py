"""Reciprocal Rank Fusion (RRF) for merging ranked result lists.

RRF formula (Cormack et al., 2009):
    score(d) = sum over rankers of 1 / (k + rank(d))

k=60 is the conventional constant — it dampens the impact of very high ranks
and prevents a single top-1 result from dominating the fused score.

Each ranker contributes an independent ranked list.  Documents that appear in
multiple lists accumulate score from each.  The fused list is sorted descending
by total RRF score.

This module is intentionally pure Python with no external dependencies so it
can be unit-tested without a database or embedding model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
RRF_K = 60  # conventional constant; rarely needs tuning


@dataclass
class RankedResult:
    """One item in a fused result list."""
    chunk_id: str
    rrf_score: float
    # Payload forwarded from whichever retriever first supplied this chunk.
    payload: dict = field(default_factory=dict)


def reciprocal_rank_fusion(
    *ranked_lists: list[dict],
    id_key: str = "chunk_id",
    k: int = RRF_K,
) -> list[RankedResult]:
    """Fuse one or more ranked lists via Reciprocal Rank Fusion.

    Parameters
    ----------
    *ranked_lists:
        Each list is a sequence of dicts already ordered best-first.
        Every dict must contain the key given by ``id_key``.
    id_key:
        The field used to identify and deduplicate results across lists.
    k:
        RRF smoothing constant (default 60).

    Returns
    -------
    List of :class:`RankedResult` sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            cid = item[id_key]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            # Keep the first payload we see for this chunk_id; all rankers
            # return the same underlying row so any copy is fine.
            if cid not in payloads:
                payloads[cid] = {k_: v for k_, v in item.items() if k_ != id_key}

    return sorted(
        [
            RankedResult(chunk_id=cid, rrf_score=score, payload=payloads[cid])
            for cid, score in scores.items()
        ],
        key=lambda r: r.rrf_score,
        reverse=True,
    )