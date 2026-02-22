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
    s = _ZW.sub("", s)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = _DASH.sub("-", s)
    s = s.replace("\u00a0", " ")
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


def _best_substring_snap(raw_paragraph: str, raw_quote: str) -> str | None:
    """
    Return a verbatim substring from raw_paragraph that best matches raw_quote.

    Key feature: handles PDF extraction artifacts like:
      - intra-word spaces ("sen d" -> "send")
      - spaces around hyphens ("e -mail" -> "e-mail")
      - NBSP / weird whitespace collapsing
      - curly quotes / odd dashes normalization for matching only

    Implementation:
      - Build a normalized paragraph string WITH a mapping from normalized indices -> raw indices.
      - Normalize the quote similarly (without mapping).
      - Find the normalized quote inside normalized paragraph; map back to raw substring.
    """

    if not raw_paragraph or not raw_quote:
        return None

    def _canon_char(c: str) -> str:
        # Normalize visually equivalent punctuation for MATCHING only.
        # Mapping back uses raw indices, so verbatim is preserved in output.
        if c in ("\u2018", "\u2019", "\u2032"):
            return "'"
        if c in ("\u201C", "\u201D", "\u2033"):
            return '"'
        if c in ("\u2013", "\u2014", "\u2212"):
            return "-"
        if c == "\u00A0":  # NBSP
            return " "
        if c == "\u00AD":  # soft hyphen
            return ""
        return c

    def _is_ws(c: str) -> bool:
        return c.isspace() or c == "\u00A0"

    def _next_non_ws(s: str, i: int) -> tuple[int, str] | None:
        n = len(s)
        j = i
        while j < n and _is_ws(s[j]):
            j += 1
        if j >= n:
            return None
        return j, _canon_char(s[j])

    def _norm_with_map(raw: str) -> tuple[str, list[int]]:
        """
        Returns:
          norm_str: normalized string used for matching
          norm2raw: list where norm2raw[k] = raw_index of the raw char that produced norm_str[k]
        """
        norm_chars: list[str] = []
        norm2raw: list[int] = []

        n = len(raw)
        i = 0

        # Track last emitted non-space char to decide whether to drop intra-word spaces.
        last_emitted: str | None = None

        while i < n:
            c_raw = raw[i]

            # whitespace run
            if _is_ws(c_raw):
                nxt = _next_non_ws(raw, i)
                # Find the end of this whitespace run
                j = i
                while j < n and _is_ws(raw[j]):
                    j += 1

                if nxt is None:
                    # trailing whitespace -> ignore
                    i = j
                    continue

                _, nxt_c = nxt
                prev = last_emitted

                # Drop whitespace if it appears to be inside a word: "sen d" -> "send"
                # Also drop around hyphens: "e -mail" or "e- mail" -> "e-mail"
                if prev and prev.isalnum() and nxt_c and nxt_c.isalnum():
                    i = j
                    continue
                if (prev == "-" and nxt_c) or (nxt_c == "-" and prev):
                    i = j
                    continue

                # Otherwise collapse to single space (but avoid double spaces)
                if norm_chars and norm_chars[-1] != " ":
                    norm_chars.append(" ")
                    norm2raw.append(i)  # map space to first ws char of the run
                    last_emitted = " "
                i = j
                continue

            # non-whitespace char
            c = _canon_char(c_raw)
            if c == "":
                i += 1
                continue

            # Lower-case for matching
            c_low = c.lower()
            norm_chars.append(c_low)
            norm2raw.append(i)
            last_emitted = c_low
            i += 1

        # Trim leading/trailing spaces in normalized form (adjust mapping accordingly)
        # Leading
        while norm_chars and norm_chars[0] == " ":
            norm_chars.pop(0)
            norm2raw.pop(0)
        # Trailing
        while norm_chars and norm_chars[-1] == " ":
            norm_chars.pop()
            norm2raw.pop()

        return "".join(norm_chars), norm2raw

    def _norm_quote(raw: str) -> str:
        # Quote normalization without mapping (same rules as paragraph).
        norm, _ = _norm_with_map(raw)
        return norm

    para_norm, norm2raw = _norm_with_map(raw_paragraph)
    quote_norm = _norm_quote(raw_quote)

    if not para_norm or not quote_norm:
        return None

    # Direct normalized substring search
    pos = para_norm.find(quote_norm)
    if pos != -1:
        start_norm = pos
        end_norm = pos + len(quote_norm)

        # Map normalized span -> raw span
        if start_norm < 0 or end_norm > len(norm2raw) or start_norm >= end_norm:
            return None

        start_raw = norm2raw[start_norm]
        end_raw_inclusive = norm2raw[end_norm - 1]
        raw_slice = raw_paragraph[start_raw : end_raw_inclusive + 1]

        # Sanity: must be an actual substring of the raw paragraph (always true by construction)
        if raw_slice and raw_slice in raw_paragraph:
            return raw_slice

    # If we can't find it, return None and let upstream decide (semantic search / drop, etc.)
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
            # side cache (optional)
            if not hasattr(self, "_query_cache_scores"):
                setattr(self, "_query_cache_scores", {})
            self._query_cache_scores[key] = []
            return []

        V = self._vecs
        if V.size == 0:
            self._query_cache[key] = []
            if not hasattr(self, "_query_cache_scores"):
                setattr(self, "_query_cache_scores", {})
            self._query_cache_scores[key] = []
            return []

        # Vectorized cosine similarity:
        # sims[i] = dot(qv, V[i]) / (||qv|| * ||V[i]||)
        q_norm = float(np.linalg.norm(qv))
        if q_norm <= 0.0:
            self._query_cache[key] = []
            if not hasattr(self, "_query_cache_scores"):
                setattr(self, "_query_cache_scores", {})
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

        # Side-cache scores (optional but useful for thresholding later)
        if not hasattr(self, "_query_cache_scores"):
            setattr(self, "_query_cache_scores", {})
        self._query_cache_scores[key] = [(i, float(sims[i])) for i in idxs]

        return idxs

    
    def snap_anchor(self, anchor, para_map, allowed_para_ids=None):
        """
        Attempt to 'snap' anchor.quote to an exact verbatim substring inside an allowed paragraph.

        - First try within anchor.para_id (if allowed).
        - If that fails, use semantic search over allowed paras to pick candidates, then try snapping there.
        - Returns a NEW anchor with corrected (para_id, quote) if successful, else None.
        """
        if not anchor:
            return None

        pid = getattr(anchor, "para_id", None)
        q = getattr(anchor, "quote", None) or ""

        if not q or not isinstance(q, str):
            return None

        if allowed_para_ids is None:
            allowed_para_ids = set(para_map.keys())
        else:
            allowed_para_ids = set(allowed_para_ids)

        # 1) Try snapping inside same paragraph (if allowed)
        if pid in allowed_para_ids:
            raw = para_map.get(pid, "") or ""
            snapped = _best_substring_snap(raw, q)
            if snapped:
                try:
                    return dc_replace(anchor, quote=snapped)
                except Exception:
                    try:
                        anchor.quote = snapped
                        return anchor
                    except Exception:
                        return None

        # 2) Semantic fallback: search among allowed paras and try snapping there
        # NOTE: semantic search can be strict; snapping is the gate.
        try:
            candidates = self.search(
                query=q,
                allowed_para_ids=allowed_para_ids,
                top_k=5,
                min_score=0.62,
            )
        except Exception:
            candidates = []

        for pid2, score in (candidates or []):
            if pid2 not in allowed_para_ids:
                continue
            raw2 = para_map.get(pid2, "") or ""
            snapped2 = _best_substring_snap(raw2, q)
            if not snapped2:
                continue

            try:
                return dc_replace(anchor, para_id=pid2, quote=snapped2)
            except Exception:
                try:
                    anchor.para_id = pid2
                    anchor.quote = snapped2
                    return anchor
                except Exception:
                    return None

        return None


    def repair_verdict_anchors(self, verdict, para_map, mode: str = "A"):
        """
        Repair verdict anchors so that each anchor.quote is VERBATIM (substring) of para_map[para_id].

        Modes:
        - "A" (default): try to snap; if an anchor cannot be repaired verbatim, DROP it.
                        (prevents one bad anchor from poisoning the whole run)
        - "B" (stricter): if ANY anchor cannot be repaired, return verdict unchanged (current behavior).

        Returns:
        (verdict2, repaired_count)
            repaired_count counts anchors that were changed OR dropped (mode A).
        """
        anchors = getattr(verdict, "anchors", None) or []
        if not anchors:
            return verdict, 0

        mode = (mode or "A").strip().upper()
        if mode not in ("A", "B"):
            mode = "A"

        allowed_para_ids = set(para_map.keys())

        repaired = 0
        new_anchors = []
        failed_any = False

        for a in anchors:
            a2 = self.snap_anchor(a, para_map, allowed_para_ids=allowed_para_ids)

            if a2 is None:
                failed_any = True
                if mode == "A":
                    # Drop the anchor (do NOT keep invalid anchors)
                    repaired += 1
                    continue
                else:
                    # Mode B: keep as-is; this will likely fail validation upstream
                    new_anchors.append(a)
                    continue

            # Anchor repaired (or at least snapped to verbatim)
            try:
                if a2 != a:
                    repaired += 1
            except Exception:
                # If objects aren't comparable, count it as repaired when snapping succeeded.
                repaired += 1

            new_anchors.append(a2)

        if mode == "B" and failed_any:
            # Strict mode: refuse partial repairs; keep original to force rejection upstream
            return verdict, 0

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