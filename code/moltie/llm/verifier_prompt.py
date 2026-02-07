from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..schemas.query_object import AtomQuery
from ..schemas.run_config import RunConfig

# Evidence pack shape expected by the verifier:
# evidence_pack = {
#   "doc_id": str,
#   "doc_meta": dict,
#   "paras": [{"para_id": "p00012", "text": "..."}, ...]   # already retrieved/filtered
#   "retrieval": {"method": "...", "score": 12.3}         # optional
# }

_VERDICT_SCHEMA_TEXT = r"""
Return STRICT JSON ONLY (no markdown, no commentary). Must be a single JSON object.

Schema:
{
  "atom_id": "string",
  "doc_id": "string",
  "relevant": true|false,
  "matched_X": ["X1","X2",...],
  "precedent_score": 0-100,
  "confidence": 0-100,
  "anchors": [
    {"para_id": "p00001", "quote": "verbatim substring from provided paras", "why_it_matters": "short"}
  ],
  "use_mode": "support|contrast|harmful",
  "proposition_winner": "claimant|respondent|mixed|unclear",
  "appeal_outcome": "allowed|dismissed|remitted|mixed|unknown",
  "successful_party": "claimant|respondent|mixed|unclear",
  "distinguishers": ["..."],
  "note": "ONE sentence",
  "retrieval_score": number|null,
  "retrieval_method": "string|null"
}

Hard rules:
- anchors MUST use para_id values that exist in the provided paras list.
- each anchor.quote MUST be a verbatim substring of the corresponding para text you were given.
- If you cannot provide >= anchors_required strong anchors, set relevant=false and keep precedent_score <= 40.
- Do NOT invent facts, para numbers, parties, outcomes, or quotes.
- If the case is relevant but overall supports the respondent against the proposition, mark use_mode="harmful".
- "contrast" means it states the proper standard / test but does not clearly condemn misconduct.
"""


def build_verifier_prompt(
    atom: AtomQuery,
    evidence_pack: Dict[str, Any],
    cfg: RunConfig,
) -> str:
    """
    Creates a single-shot verifier prompt to produce a Verdict.
    The agent controls looping; the LLM only decides on the provided evidence.
    """
    doc_id = str(evidence_pack.get("doc_id") or "").strip()
    doc_meta = evidence_pack.get("doc_meta") or {}
    paras = evidence_pack.get("paras") or []
    retrieval = evidence_pack.get("retrieval") or {}

    # Keep payload tight (avoid sending huge docs)
    # Paras should already be top-K from corpus retrieval.
    paras_compact = [
        {"para_id": str(p.get("para_id") or "").strip(), "text": str(p.get("text") or "").strip()}
        for p in paras
        if str(p.get("para_id") or "").strip() and str(p.get("text") or "").strip()
    ]

    atom_compact = {
        "atom_id": atom.atom_id,
        "x_tests": atom.x_tests,
        "proposition": atom.proposition,
        "positive_indicators": atom.positive_indicators,
        "excludes": atom.excludes,
        "keyword_seeds": atom.keyword_seeds,
        "expansion_terms": atom.expansion_terms,
    }

    # The model needs explicit instruction: it only sees these paras; anchors must come from them.
    payload = {
        "atom": atom_compact,
        "doc_id": doc_id,
        "doc_meta": doc_meta,
        "anchors_required": cfg.anchors_required,
        "paras": paras_compact,
        "retrieval": retrieval,
    }

    return (
        "You are a legal precedent verifier.\n"
        "Your job: decide whether the provided appeal evidence supports the given proposition.\n"
        "You MUST follow the schema and hard rules.\n\n"
        f"{_VERDICT_SCHEMA_TEXT}\n\n"
        "Input JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
    )
