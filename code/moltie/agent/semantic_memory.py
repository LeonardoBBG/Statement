# moltie/agent/semantic_memory.py
from __future__ import annotations

import math
import re
import hashlib
from dataclasses import dataclass, replace as dc_replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_WS = re.compile(r"\s+")
_ZW = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_DASH = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")


def _norm(s: str) -> str:
    if not s:
        return ""

    # Remove zero-width chars
    s = _ZW.sub("", s)

    # Normalize quotes
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # Normalize dashes
    s = _DASH.sub("-", s)

    # NBSP
    s = s.replace("\u00a0", " ")

    # --- CRITICAL FIXES BELOW ---

    # 1) Fix spaced hyphens: "e -mail" -> "e-mail"
    s = re.sub(r"\s*-\s*", "-", s)

    # 2) Fix broken intra-word splits: "sen d" -> "send"
    # Only merge if both sides are letters and it's a single space
    s = re.sub(r"\b([A-Za-z]) ([A-Za-z])\b", r"\1\2", s)

    # 3) Collapse whitespace
    s = _WS.sub(" ", s).strip()

    return s


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _sentenceish_split(text: str, max_chunk_chars: int = 500) -> List[str]:
    """
    Conservative splitter. We do NOT want clever; we want stable, repeatable chunks.
    """
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[\.\?\!;:])\s+", t)
    out: List[str] = []
    buf: List[str] = []
    size = 0
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if size + len(p) + 1 > max_chunk_chars and buf:
            out.append(" ".join(buf).strip())
            buf = [p]
            size = len(p)
        else:
            buf.append(p)
            size += len(p) + 1
    if buf:
        out.append(" ".join(buf).strip())
    return out


def _best_substring_snap(
    quote: str,
    para_text: str,
    *,
    max_window_chars: int = 900,
    stride_chars: int = 60,
    max_return_chars: int = 240,
) -> Optional[str]:
    """
    Deterministic "snap-to-verbatim":
    Find a verbatim substring from para_text whose normalized form best contains the normalized quote,
    or best overlaps it.

    Returns a verbatim substring from the ORIGINAL para_text, or None.
    """
    if not quote or not para_text:
        return None

    # Fast path: exact
    if quote in para_text:
        return quote

    nq = _norm(quote)
    nt = _norm(para_text)
    if not nq or not nt:
        return None

    # Fast path: normalized containment
    if nq in nt:
        best = None
        best_len = None

        raw = para_text
        L = len(raw)
        win = min(max_window_chars, max(200, len(quote) + 120))
        stride = max(20, stride_chars)

        for s in range(0, max(1, L), stride):
            e = min(L, s + win)
            w = raw[s:e]
            if nq in _norm(w):
                cur_len = len(w)
                if best is None or cur_len < best_len:
                    best = w
                    best_len = cur_len
            if e >= L:
                break

        if best is not None:
            best = best.strip()
            if len(best) <= max_return_chars:
                return best
            return None

    # Fallback: token overlap scoring on windows (still deterministic)
    q_toks = set(_norm(quote).lower().split())
    if not q_toks:
        return None

    raw = para_text
    L = len(raw)
    win = min(max_window_chars, max(240, len(quote) + 160))
    stride = max(20, stride_chars)

    best_w = None
    best_score = 0.0

    for s in range(0, max(1, L), stride):
        e = min(L, s + win)
        w = raw[s:e]
        w_toks = set(_norm(w).lower().split())
        if not w_toks:
            continue
        inter = len(q_toks & w_toks)
        score = inter / max(1, len(q_toks))
        if score > best_score:
            best_score = score
            best_w = w
        if e >= L:
            break

    if best_w is not None and best_score >= 0.85:
        best_w = best_w.strip()
        if len(best_w) <= max_return_chars:
            return best_w

    return None


# ---------------------------
# Embedding providers (pluggable)
# ---------------------------

class Embedder:
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class TfidfEmbedder(Embedder):
    """
    No external deps, fast enough, surprisingly strong for legal text.
    """
    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}

    def _tokenize(self, s: str) -> List[str]:
        s = _norm(s).lower()
        return [t for t in re.split(r"[^a-z0-9]+", s) if t]

    def fit(self, texts: List[str]) -> None:
        vocab: Dict[str, int] = {}
        for t in texts:
            for tok in set(self._tokenize(t)):
                vocab[tok] = vocab.get(tok, 0) + 1
        toks = [k for k, c in vocab.items() if c >= 2]
        self._vocab = {tok: i for i, tok in enumerate(sorted(toks))}

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._vocab:
            return np.zeros((len(texts), 1), dtype=np.float32)

        X = np.zeros((len(texts), len(self._vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = self._tokenize(t)
            if not toks:
                continue
            tf: Dict[int, int] = {}
            for tok in toks:
                j = self._vocab.get(tok)
                if j is None:
                    continue
                tf[j] = tf.get(j, 0) + 1
            if not tf:
                continue
            for j, c in tf.items():
                X[i, j] = math.log(1.0 + float(c))
        return X


@dataclass(frozen=True)
class MemoryItem:
    para_id: str
    chunk_text: str
    chunk_hash: str


def _anchor_get(a: Any, k: str) -> Any:
    return getattr(a, k) if hasattr(a, k) else (a.get(k) if isinstance(a, dict) else None)


def _anchor_set_dict(a: dict, para_id: str, quote: str) -> dict:
    b = dict(a)
    b["para_id"] = para_id
    b["quote"] = quote
    return b


class SemanticMemory:
    """
    Per-doc semantic memory:
    - stable chunks from paras
    - vector index (optional but recommended)
    - snap-to-verbatim utility
    """

    def __init__(
        self,
        *,
        embedder: Optional[Embedder] = None,
        max_chunk_chars: int = 500,
        top_k: int = 8,
        prefer_same_para: bool = True,
    ) -> None:
        self.embedder = embedder or TfidfEmbedder()
        self.max_chunk_chars = int(max_chunk_chars)
        self.top_k = int(top_k)
        self.prefer_same_para = bool(prefer_same_para)

        self.items: List[MemoryItem] = []
        self._para_text: Dict[str, str] = {}
        self._vecs: Optional[np.ndarray] = None

        # cache: quote_hash -> list of candidate item indices
        self._query_cache: Dict[str, List[int]] = {}
        self._query_cache_scores: Dict[str, List[Tuple[int, float]]] = {}

    
    def build(self, paras: List[Dict[str, str]]) -> None:
        self.items = []
        self._para_text = {p["para_id"]: (p.get("text") or "") for p in (paras or []) if p.get("para_id")}
        chunks: List[str] = []

        seen = set()
        for pid, txt in self._para_text.items():
            for ch in _sentenceish_split(txt, max_chunk_chars=self.max_chunk_chars):
                key = _h(pid + "::" + ch)
                if key in seen:
                    continue
                seen.add(key)
                self.items.append(MemoryItem(para_id=pid, chunk_text=ch, chunk_hash=key))
                chunks.append(ch)

        if isinstance(self.embedder, TfidfEmbedder):
            self.embedder.fit(chunks)

        self._vecs = self.embedder.embed(chunks) if chunks else None
        self._query_cache.clear()
        self._query_cache_scores.clear()

    def _query_indices(self, text: str) -> List[int]:
        """
        Return top-k candidate chunk indices for the given query text.

        Keeps existing public contract: returns List[int].

        Also caches similarity scores for the returned indices in a side-cache:
            self._query_cache_scores[key] = List[Tuple[int, float]]
        so downstream logic can apply thresholds without recomputing sims.
        """
        if not self.items or self._vecs is None:
            return []

        key = _h(text)
        if key in self._query_cache:
            return self._query_cache[key]

        # Embed query
        qv = self.embedder.embed([text])[0]
        if qv is None:
            self._query_cache[key] = []
            self._query_cache_scores[key] = []
            return []

        V = self._vecs
        if V.size == 0:
            self._query_cache[key] = []
            self._query_cache_scores[key] = []
            return []

        # Vectorized cosine similarity:
        # sims[i] = dot(qv, V[i]) / (||qv|| * ||V[i]||)
        q_norm = float(np.linalg.norm(qv))
        if q_norm <= 0.0:
            self._query_cache[key] = []
            self._query_cache_scores[key] = []
            return []

        v_norms = np.linalg.norm(V, axis=1)
        denom = (q_norm * v_norms)
        denom = np.where(denom > 0.0, denom, 1.0)  # avoid div-by-zero
        sims = (V @ qv) / denom

        # Top-k indices
        k = max(1, int(self.top_k))
        if sims.shape[0] <= k:
            idxs = list(range(int(sims.shape[0])))
            idxs.sort(key=lambda i: float(sims[i]), reverse=True)
        else:
            # argpartition for speed, then sort those k
            top = np.argpartition(-sims, kth=k - 1)[:k]
            idxs = [int(i) for i in top]
            idxs.sort(key=lambda i: float(sims[i]), reverse=True)

        self._query_cache[key] = idxs

        self._query_cache_scores[key] = [(i, float(sims[i])) for i in idxs]

        return idxs

    
    def snap_anchor(
        self,
        *,
        para_id: str,
        quote: str,
        allowed_para_ids: Optional[set[str]] = None,
    ) -> Optional[Tuple[str, str]]:
        """
        Returns (para_id, verbatim_quote) or None.

        Strategy:
        1) Enforce allowed_para_ids (hard gate).
        2) Try snap within the provided para_id first.
        3) If not found, query semantic memory for similar chunks.
        - Optionally prefer same para_id ordering.
        - Apply a similarity threshold when crossing to other para_ids to prevent drift.
        4) For each candidate para, attempt deterministic snap-to-verbatim.

        Note: Requires _query_indices() to have populated self._query_cache_scores (optional).
        If scores cache is absent, we conservatively allow the search but still rely on snap-to-verbatim.
        """
        if not quote:
            return None

        pid0 = (para_id or "").strip()

        # Hard gate: if caller specified allowed ids and the requested para isn't allowed, don't move.
        if allowed_para_ids is not None and pid0 and pid0 not in allowed_para_ids:
            return None

        # 1) Same-para snap first (best precision)
        if pid0:
            para_txt = self._para_text.get(pid0, "")
            snapped = _best_substring_snap(quote, para_txt)
            if snapped:
                return (pid0, snapped)

        # 2) Semantic candidates
        idxs = self._query_indices(quote)
        if not idxs:
            return None

        # Pull scores populated by _query_indices
        key = _h(quote)
        scored: List[Tuple[int, float]] = list(self._query_cache_scores.get(key, []))

        score_map: Dict[int, float] = {i: s for i, s in scored} if scored else {}

        # Similarity threshold for crossing para boundaries.
        # Keep this strict because anchor repair must not become semantic substitution.
        cross_para_min_sim = 0.90

        # Order indices: prefer same para_id if requested, then by similarity (if available), then stable index.
        def _rank(i: int) -> Tuple[int, float, int]:
            same = 0 if (self.prefer_same_para and pid0 and self.items[i].para_id == pid0) else 1
            sim = score_map.get(i, 0.0)
            return (same, -sim, i)

        ordered = sorted(idxs, key=_rank)

        for i in ordered:
            it = self.items[i]
            pid2 = it.para_id

            if allowed_para_ids is not None and pid2 not in allowed_para_ids:
                continue

            # If we're moving to another paragraph, require similarity threshold when available.
            if pid0 and pid2 != pid0 and score_map:
                if score_map.get(i, 0.0) < cross_para_min_sim:
                    continue

            txt2 = self._para_text.get(pid2, "")
            snapped2 = _best_substring_snap(quote, txt2)
            if snapped2:
                return (pid2, snapped2)

        return None


    def repair_verdict_anchors(
        self,
        verdict: Any,
        *,
        allowed_para_ids: Optional[set[str]] = None,
    ) -> Tuple[Any, int]:
        """
        Returns (new_verdict, repaired_count).
        Works with frozen dataclass anchors by rebuilding the anchors list.

        Rules:
        - If quote is already a strict verbatim substring of the raw paragraph text, keep it.
        - If quote is "close" under normalization, attempt snap-to-verbatim and replace quote with verbatim slice.
        - If quote is not verbatim and not close, still attempt snap-to-verbatim.
        - allowed_para_ids is a hard constraint: do not move to disallowed paras.
        """
        anchors = getattr(verdict, "anchors", None) or []
        if not anchors:
            return verdict, 0

        repaired = 0
        new_anchors: List[Any] = []

        for a in anchors:
            pid = _anchor_get(a, "para_id")
            q = _anchor_get(a, "quote")

            if not pid or not q:
                new_anchors.append(a)
                continue

            pid_s = str(pid).strip()
            q_s = str(q)

            # Hard gate: if allowed_para_ids is specified and the given pid isn't allowed, we cannot repair it.
            # (We keep as-is; validator can reject later.)
            if allowed_para_ids is not None and pid_s and pid_s not in allowed_para_ids:
                new_anchors.append(a)
                continue

            txt = self._para_text.get(pid_s, "")

            # 1) Strict verbatim: keep
            if q_s and txt and (q_s in txt):
                new_anchors.append(a)
                continue

            # 2) If it's close under normalization, we MUST snap to verbatim (do not accept model text).
            close_norm = bool(_norm(q_s)) and bool(_norm(txt)) and (_norm(q_s) in _norm(txt))

            # 3) Attempt snap-to-verbatim (same para first; then semantic candidates if needed)
            got = self.snap_anchor(
                para_id=pid_s,
                quote=q_s,
                allowed_para_ids=allowed_para_ids,
            )

            if not got:
                # If we cannot snap to a short verbatim quote, keep as-is and let downstream
                # validation reject it rather than silently broadening the evidence span.
                new_anchors.append(a)
                continue

            pid2, q2 = got

            # Only count as repair if we actually changed the quote or para_id OR it was a close_norm case.
            changed = (pid2 != pid_s) or (q2 != q_s) or close_norm

            if isinstance(a, dict):
                new_anchors.append(_anchor_set_dict(a, pid2, q2))
                if changed:
                    repaired += 1
            else:
                try:
                    new_anchors.append(dc_replace(a, para_id=pid2, quote=q2))
                    if changed:
                        repaired += 1
                except Exception:
                    new_anchors.append(a)

        try:
            verdict2 = dc_replace(verdict, anchors=new_anchors)
        except Exception:
            try:
                verdict.anchors = new_anchors
                verdict2 = verdict
            except Exception:
                verdict2 = verdict
                repaired = 0

        return verdict2, repaired
