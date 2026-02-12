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
Return STRICT JSON ONLY. No markdown. No commentary. One JSON object.

ALLOWED KEYS ONLY (exactly these keys, no extras):
atom_id, doc_id, relevant, matched_X, precedent_score, confidence,
anchors, use_mode, proposition_winner, appeal_outcome, successful_party,
distinguishers, note, retrieval_score, retrieval_method

Schema:
{
  "atom_id": "string",
  "doc_id": "string",
  "relevant": true|false,
  "matched_X": ["X1","X2","X3","X4","X5"],
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
- anchors[].para_id MUST be one of the provided para_id values.
- anchors[].quote MUST be a verbatim substring of that paragraph text.
- If you cannot provide >= anchors_required anchors, set relevant=false and anchors=[] and precedent_score<=40.
- If relevant=false then use_mode MUST be "contrast".
- Do NOT invent parties/outcomes: if not in evidence, use "unclear"/"unknown".
- precedent_score and confidence MUST be integers in the range 0..100 (never negative). If unknown, use 0.
- note MUST be a single short sentence (if nothing to add, use "No relevant information found.").
"""


def build_verifier_prompt(
    atom: AtomQuery,
    evidence_pack: Dict[str, Any],
    cfg: RunConfig,
) -> str:
    """
    Creates a single-shot verifier prompt to produce a Verdict.
    The agent controls looping; the LLM only decides on the provided evidence.

    If cfg.harvest_mode=True, the LLM behaves like a junior "highlighter":
      - it scans this small window only
      - it extracts ONLY strong anchored hits
      - it avoids guessing outcomes/winners
    """
    doc_id = str(evidence_pack.get("doc_id") or "").strip()
    doc_meta = evidence_pack.get("doc_meta") or {}
    paras = evidence_pack.get("paras") or []
    retrieval = evidence_pack.get("retrieval") or {}

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

    harvest_mode = bool(getattr(cfg, "harvest_mode", False))

    # In harvest mode we intentionally require fewer anchors (we're scanning windows),
    # and we cap how many anchors we want back to keep memory "strong only".
    anchors_required = int(getattr(cfg, "harvest_min_anchors", cfg.anchors_required if hasattr(cfg, "anchors_required") else 1)) if harvest_mode else cfg.anchors_required
    anchors_max = 2 if harvest_mode else 4

    payload = {
        "atom": atom_compact,
        "doc_id": doc_id,
        "doc_meta": doc_meta,
        "anchors_required": anchors_required,
        "anchors_max": anchors_max,
        "paras": paras_compact,
        "retrieval": retrieval,
        "mode": "harvest" if harvest_mode else "verify",
    }

    verdict_template = {
        "atom_id": atom.atom_id,
        "doc_id": doc_id,
        "relevant": False,
        "matched_X": [],
        "precedent_score": 0,
        "confidence": 0,
        "anchors": [],
        "use_mode": "contrast",
        "proposition_winner": "unclear",
        "appeal_outcome": "unknown",
        "successful_party": "unclear",
        "distinguishers": [],
        "note": "",
        "retrieval_score": None,
        "retrieval_method": None,
    }

    harvest_instructions = ""
    if harvest_mode:
        harvest_instructions = (
            "\nHARVEST MODE (junior highlighter):\n"
            "- You are scanning a SMALL window of the document, not the whole judgment.\n"
            "- Do NOT attempt to decide the entire case.\n"
            "- Mark relevant=true ONLY if you can extract strong, clearly supportive anchored quotes.\n"
            f"- Return at most anchors_max={anchors_max} anchors. Prefer the strongest 1–2.\n"
            "- If the signal is weak/ambiguous, set relevant=false.\n"
            "- IMPORTANT: Do NOT guess appeal outcome or winners from partial text; use 'unknown'/'unclear' unless explicitly stated in these paras.\n"
        )

    return (
        "SYSTEM / OUTPUT CONTRACT (NON-NEGOTIABLE):\n"
        "1) Output ONE JSON object ONLY that matches the schema below.\n"
        "2) Use ONLY the allowed keys. If you output any other key, it is WRONG.\n"
        "3) Do NOT echo the Input JSON. Do NOT return the evidence paragraphs.\n"
        "4) Every anchor.quote must be verbatim from the provided paras, and anchor.para_id must be one of the provided para_id.\n"
        f"5) If you cannot meet anchors_required={anchors_required}, set relevant=false, anchors=[], precedent_score<=40, and use_mode=\"contrast\".\n"
        f"6) If you set relevant=true, you MUST return 1..{anchors_max} anchors (never 0).\n"
        + harvest_instructions +
        "\nYou are a legal precedent verifier.\n"
        "Your job: decide whether the provided appeal evidence supports the given proposition.\n"
        "You MUST follow the schema and hard rules.\n\n"
        f"{_VERDICT_SCHEMA_TEXT}\n\n"
        "OUTPUT JSON TEMPLATE (fill values, keep keys identical, output ONLY this object):\n"
        f"{json.dumps(verdict_template, ensure_ascii=False)}\n\n"
        "Input JSON (DO NOT ECHO THIS BACK; use it only to make the decision):\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
    )
