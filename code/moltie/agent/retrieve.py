from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Set

from ..schemas.query_object import AtomQuery

_WORD_RE = re.compile(r"[a-z]{3,}", re.IGNORECASE)

def tokenize(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())

@dataclass(frozen=True)
class RetrievalResult:
    doc_id: str
    paras: List[Dict[str, str]]          # [{"para_id":..., "text":...}, ...]
    retrieval: Dict[str, Any]            # {"method": "...", "score": float, "matched_paras": int}


def _para_hits(p: Dict[str, str], seed_tokens: Set[str]) -> int:
    tks = set(tokenize(p.get("text", "")))
    return len(seed_tokens.intersection(tks))


def retrieve_windowed_evidence(
    doc_id: str,
    paras: List[Dict[str, str]],
    atom: AtomQuery,
    *,
    k: int = 12,
    min_hits: int = 2,
    window_size: int = 24,
    stride: int = 12,
    top_windows: int = 3,
    visited_starts: Optional[Set[int]] = None,
    ensure_coverage: bool = True,
) -> RetrievalResult:
    """
    Sliding-window evidence retrieval (coverage-aware).

    - Score paragraphs by token overlap (same as MVP)
    - Scan windows across the full doc
    - Select top-N windows (optionally skipping visited windows)
    - Merge their paragraphs and return up to k (ranked)
    - If no hits anywhere, return stratified windows (start/mid/end)

    This prevents "only first pages" and replaces 'mid_fallback' with real coverage.
    """
    n = len(paras)
    if n == 0:
        return RetrievalResult(doc_id=doc_id, paras=[], retrieval={
            "method": "empty_doc",
            "score": 0.0,
            "matched_paras": 0,
            "k": k,
            "min_hits": min_hits,
            "window_size": window_size,
            "stride": stride,
        })

    seed_text = " ".join([atom.proposition] + list(atom.positive_indicators or []))
    seed_tokens = set(tokenize(seed_text))

    visited_starts = visited_starts or set()

    # 1) Window scan
    window_scores: List[Tuple[int, int, int]] = []  # (score, start, end)
    total_matching_paras = 0

    # Precompute per-para hits once (cheap + reuse)
    hits_by_idx = [ _para_hits(paras[i], seed_tokens) for i in range(n) ]
    total_matching_paras = sum(1 for h in hits_by_idx if h >= min_hits)

    # Score windows by sum of (hits) for paras meeting min_hits
    for start in range(0, n, stride):
        if start in visited_starts:
            continue
        end = min(n, start + window_size)
        if end <= start:
            continue
        score = 0
        for i in range(start, end):
            h = hits_by_idx[i]
            if h >= min_hits:
                score += h
        window_scores.append((score, start, end))
        if end == n:
            break

    window_scores.sort(key=lambda x: x[0], reverse=True)

    # 2) Pick top windows with positive score
    picked = [w for w in window_scores if w[0] > 0][:top_windows]

    # 3) If none, stratified coverage fallback (start+mid+end)
    if not picked:
        def clip_window(center: int) -> Tuple[int, int]:
            s = max(0, center - window_size // 2)
            e = min(n, s + window_size)
            s = max(0, e - window_size)  # re-adjust
            return s, e

        centers = []
        if ensure_coverage:
            centers = [0, n // 2, max(0, n - 1)]
        else:
            centers = [n // 2]

        spans = []
        for c in centers:
            s, e = clip_window(c)
            spans.append((s, e))

        # Deduplicate spans
        uniq_spans = []
        seen = set()
        for s, e in spans:
            key = (s, e)
            if key not in seen:
                seen.add(key)
                uniq_spans.append((s, e))

        pack: List[Dict[str, str]] = []
        for s, e in uniq_spans:
            pack.extend(paras[s:e])

        # Cap to k by keeping original order (coverage pack)
        pack = pack[:k]

        return RetrievalResult(
            doc_id=doc_id,
            paras=pack,
            retrieval={
                "method": "coverage_fallback",
                "score": 0.0,
                "matched_paras": total_matching_paras,
                "k": k,
                "min_hits": min_hits,
                "window_size": window_size,
                "stride": stride,
                "top_windows": top_windows,
                "picked_windows": [{"start": s, "end": e, "score": 0} for s, e in uniq_spans],
            },
        )

    # 4) Merge picked windows and rank paras globally by hits
    candidate_idxs: List[int] = []
    for score, start, end in picked:
        candidate_idxs.extend(range(start, end))

    # Dedup indices while preserving order
    seen_i = set()
    uniq_idxs = []
    for i in candidate_idxs:
        if i not in seen_i:
            seen_i.add(i)
            uniq_idxs.append(i)

    # Rank within candidate set by hits desc, then keep top-k
    ranked = sorted(
        ((hits_by_idx[i], i) for i in uniq_idxs),
        key=lambda x: x[0],
        reverse=True
    )
    top_idxs = [i for h, i in ranked if h >= min_hits][:k]

    # If ranking filtered too hard (rare), relax: take best available from candidates
    if not top_idxs:
        top_idxs = [i for _, i in ranked[:k]]

    top_paras = [paras[i] for i in top_idxs]

    return RetrievalResult(
        doc_id=doc_id,
        paras=top_paras,
        retrieval={
            "method": "sliding_window",
            "score": float(sum(hits_by_idx[i] for i in top_idxs)),
            "matched_paras": total_matching_paras,
            "k": k,
            "min_hits": min_hits,
            "window_size": window_size,
            "stride": stride,
            "top_windows": top_windows,
            "picked_windows": [{"start": s, "end": e, "score": sc} for sc, s, e in picked],
        },
    )
