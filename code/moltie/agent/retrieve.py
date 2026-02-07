from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from ..schemas.query_object import AtomQuery


_WORD_RE = re.compile(r"[a-z]{3,}", re.IGNORECASE)

def tokenize(s: str) -> List[str]:
    return _WORD_RE.findall((s or "").lower())

@dataclass(frozen=True)
class RetrievalResult:
    doc_id: str
    paras: List[Dict[str, str]]          # [{"para_id":..., "text":...}, ...]
    retrieval: Dict[str, Any]            # {"method": "...", "score": float, "matched_paras": int}


def retrieve_evidence(
    doc_id: str,
    paras: List[Dict[str, str]],
    atom: AtomQuery,
    k: int = 12,
    min_hits: int = 2,
) -> RetrievalResult:
    """
    MVP evidence retrieval:
    - build token set from proposition + indicators
    - score paragraph by token overlaps
    - take top-k
    - fallback to middle chunk if no matches
    """
    seed_text = " ".join([atom.proposition] + list(atom.positive_indicators or []))
    seed_tokens = set(tokenize(seed_text))

    scored: List[Tuple[int, Dict[str, str]]] = []
    for p in paras:
        tks = set(tokenize(p.get("text", "")))
        hits = len(seed_tokens.intersection(tks))
        if hits >= min_hits:
            scored.append((hits, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [p for _, p in scored[:k]]

    if not top:
        mid = len(paras) // 2
        top = paras[max(0, mid - k//2): min(len(paras), mid + k//2)]
        method = "mid_fallback"
        score = 0.0
        matched = 0
    else:
        method = "token_overlap"
        score = float(sum(h for h, _ in scored[:k]))
        matched = len(scored)

    return RetrievalResult(
        doc_id=doc_id,
        paras=top,
        retrieval={"method": method, "score": score, "matched_paras": matched, "k": k, "min_hits": min_hits},
    )
