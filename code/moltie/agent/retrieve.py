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

def _maybe_rechunk_single_blob_paras(
    paras: List[Dict[str, str]],
    *,
    min_len: int = 40,
    max_para_chars: int = 1800,
    blob_threshold: int = 5000,
) -> List[Dict[str, str]]:
    """
    Safety guard: if upstream PDF extraction collapsed everything into ONE giant paragraph,
    split it into multiple paras so retrieval + anchors can work.
    """
    if not paras or len(paras) != 1:
        return paras

    text = (paras[0].get("text") or "").strip()
    if len(text) < blob_threshold:
        return paras

    t = text.replace("\r\n", "\n").replace("\r", "\n")

    parts = [p.strip() for p in t.split("\n\n") if p.strip()]

    if len(parts) <= 3:
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        parts = []
        buf: List[str] = []
        size = 0
        for ln in lines:
            if size + len(ln) + 1 > max_para_chars and buf:
                parts.append(" ".join(buf).strip())
                buf = [ln]
                size = len(ln)
            else:
                buf.append(ln)
                size += len(ln) + 1
        if buf:
            parts.append(" ".join(buf).strip())

    out: List[str] = []
    for p in parts:
        p = " ".join(p.split()).strip()
        if len(p) < min_len:
            continue
        if len(p) <= max_para_chars:
            out.append(p)
        else:
            start = 0
            while start < len(p):
                chunk = p[start : start + max_para_chars].strip()
                if len(chunk) >= min_len:
                    out.append(chunk)
                start += max_para_chars

    return [{"para_id": f"p{i:05d}", "text": p} for i, p in enumerate(out, start=1)]


from typing import Any, Dict, List, Optional, Set, Tuple

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
    # ---- FIX B: prefer visited_windows (start,end). keep visited_starts for backward-compat ----
    visited_windows: Optional[Set[Tuple[int, int]]] = None,
    visited_starts: Optional[Set[int]] = None,
    ensure_coverage: bool = True,
) -> RetrievalResult:
    """
    Sliding-window evidence retrieval (coverage-aware, iteration-aware).

    Fix B implemented:
    - Avoid repeating windows across iters by skipping *visited windows* (start,end) first.
    - ensure_coverage fallback prefers unvisited spans (start/mid/end) before reusing.
    - Still supports visited_starts for older callers (coarser skip).
    """
    paras = _maybe_rechunk_single_blob_paras(paras)
    n = len(paras)

    if n == 0:
        return RetrievalResult(
            doc_id=doc_id,
            paras=[],
            retrieval={
                "method": "empty_doc",
                "score": 0.0,
                "matched_paras": 0,
                "k": k,
                "min_hits": min_hits,
                "window_size": window_size,
                "stride": stride,
                "top_windows": top_windows,
                "picked_windows": [],
            },
        )

    # --- build seed tokens ---
    # Include refinement-driven recall terms so later iterations meaningfully alter retrieval.
    seed_parts = [atom.proposition]
    seed_parts.extend(list(atom.positive_indicators or []))
    seed_parts.extend(list(getattr(atom, "keyword_seeds", []) or []))
    seed_parts.extend(list(getattr(atom, "expansion_terms", []) or []))
    seed_text = " ".join([s for s in seed_parts if isinstance(s, str) and s.strip()])
    seed_tokens = set(tokenize(seed_text))

    # --- visited structures ---
    visited_windows = set(visited_windows or set())

    # If caller still passes visited_starts, treat those as "visited any span with that start".
    visited_starts = set(visited_starts or set())

    def _is_visited(start: int, end: int) -> bool:
        if (start, end) in visited_windows:
            return True
        if start in visited_starts:
            return True
        return False

    # Precompute per-para hits once (cheap + reuse)
    hits_by_idx = [_para_hits(paras[i], seed_tokens) for i in range(n)]
    total_matching_paras = sum(1 for h in hits_by_idx if h >= min_hits)

    # Build explicit window starts so we always include final (n - window_size)
    starts = list(range(0, max(1, n), max(1, stride)))
    if n > window_size:
        last_start = max(0, n - window_size)
        if last_start not in starts:
            starts.append(last_start)
    starts = sorted(set(starts))
    est_total_windows = max(1, len(starts))

    # 1) Scan windows (skip visited first)
    # store (score, start, end)
    window_scores: List[Tuple[int, int, int]] = []
    for start in starts:
        end = min(n, start + window_size)
        if end <= start:
            continue
        if _is_visited(start, end):
            continue

        score = 0
        for i in range(start, end):
            h = hits_by_idx[i]
            if h >= min_hits:
                score += h

        window_scores.append((score, start, end))

    # If everything is "visited" but we still need to do something, allow reuse
    # (but only if absolutely necessary)
    if not window_scores:
        for start in starts:
            end = min(n, start + window_size)
            if end <= start:
                continue
            score = 0
            for i in range(start, end):
                h = hits_by_idx[i]
                if h >= min_hits:
                    score += h
            window_scores.append((score, start, end))

    window_scores.sort(key=lambda x: x[0], reverse=True)

    # 2) Pick top windows with positive score
    picked = [w for w in window_scores if w[0] > 0][:max(1, top_windows)]

    # Helper: clip a centered span to valid window boundaries
    def clip_window(center: int) -> Tuple[int, int]:
        s = max(0, center - window_size // 2)
        e = min(n, s + window_size)
        s = max(0, e - window_size)  # re-adjust
        return s, e

    # Helper: coverage spans, preferring unvisited
    def build_coverage_spans() -> List[Tuple[int, int]]:
        if ensure_coverage:
            centers = [0, n // 2, max(0, n - 1)]
        else:
            centers = [n // 2]

        spans: List[Tuple[int, int]] = [clip_window(c) for c in centers]

        # de-dup while preserving order
        out: List[Tuple[int, int]] = []
        seen = set()
        for s, e in spans:
            key = (s, e)
            if key not in seen:
                seen.add(key)
                out.append(key)

        # Prefer unvisited spans first
        fresh = [se for se in out if not _is_visited(se[0], se[1])]
        if fresh:
            return fresh

        # if all are visited, return them anyway (we must return something)
        return out

    # 3) If none picked, do coverage fallback (but prefer unvisited spans)
    if not picked:
        uniq_spans = build_coverage_spans()

        pack: List[Dict[str, str]] = []
        for s, e in uniq_spans:
            pack.extend(paras[s:e])

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
                "est_total_windows": est_total_windows,
                "picked_windows": [{"start": s, "end": e, "score": 0} for (s, e) in uniq_spans],
            },
        )

    # 4) Merge picked windows and rank paragraphs by hits globally
    candidate_idxs: List[int] = []
    for score, start, end in picked:
        candidate_idxs.extend(range(start, end))

    # Dedup indices while preserving order
    seen_i: Set[int] = set()
    uniq_idxs: List[int] = []
    for i in candidate_idxs:
        if i not in seen_i:
            seen_i.add(i)
            uniq_idxs.append(i)

    ranked = sorted(
        ((hits_by_idx[i], i) for i in uniq_idxs),
        key=lambda x: x[0],
        reverse=True,
    )

    top_idxs = [i for h, i in ranked if h >= min_hits][:k]

    # If ranking filtered too hard, relax to best available candidates
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
            "est_total_windows": est_total_windows,
            "picked_windows": [{"start": s, "end": e, "score": sc} for (sc, s, e) in picked],
        },
    )
