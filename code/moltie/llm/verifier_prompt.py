
from __future__ import annotations

import json
from typing import Any, Dict

from ..schemas.query_object import AtomQuery
from ..schemas.run_config import RunConfig


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
  "matched_X": ["X1","X2","X3","X4","X5","X6"],
  "precedent_score": 0-100,
  "confidence": 0-100,
  "anchors": [
    {"para_id": "p00001", "quote": "verbatim substring", "why_it_matters": "short"}
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

Hard rules (NON-NEGOTIABLE):
- anchors[].para_id MUST be one of the provided allowed_para_ids values (exact match).
- anchors[].quote MUST be a verbatim substring of that paragraph text (copy/paste exact).
- anchors[].quote MUST be SINGLE-LINE (no newline characters) and <= 240 characters.
- anchors[].quote MUST be the MINIMUM necessary snippet (do NOT paste the entire paragraph).
- anchors[].quote MUST NOT contain markdown fences or labels like "Para:"/"Paragraph:".
- anchors[].why_it_matters MUST be SINGLE-LINE and <= 20 words.
- If relevant=true then anchors MUST be non-empty and MUST directly support the matched_X you selected.
- If you cannot provide >= anchors_required anchors, set relevant=false, anchors=[], precedent_score<=40.
- If relevant=true then use_mode MUST be "support".
- If relevant=false then use_mode MUST be "contrast".
- Do NOT invent parties/outcomes: if not explicitly stated in the provided paras, use "unclear"/"unknown".
- precedent_score and confidence MUST be integers in 0..100. If unknown, use 0.
- note MUST be a single short sentence (or "No relevant information found.").
- IMPORTANT SCOPE GUARD: Do NOT treat remedy/compensation/medical causation (ill-health, loss, Polkey, contributory fault)
  as support for X1–X3 unless AtomQuery explicitly targets REMEDY. If remedy-focused, set relevant=false.
COPY PROTOCOL (MANDATORY):
- When you create an anchor.quote, you MUST copy-paste text EXACTLY from the paragraph.
- Do NOT paraphrase. Do NOT retype from memory. Do NOT change apostrophes/quotes.
- The quote MUST appear as a contiguous substring in that paragraph.
- If you cannot comply, set relevant=false and anchors=[].

"""


def build_verifier_prompt(
    atom: AtomQuery,
    evidence_pack: Dict[str, Any],
    cfg: RunConfig,
) -> str:

    doc_id = str(evidence_pack.get("doc_id") or "").strip()
    doc_meta = evidence_pack.get("doc_meta") or {}
    paras = evidence_pack.get("paras") or []
    retrieval = evidence_pack.get("retrieval") or {}

    paras_compact = [
        {
            "para_id": str(p.get("para_id") or "").strip(),
            "text": str(p.get("text") or "").strip(),
        }
        for p in paras
        if str(p.get("para_id") or "").strip()
        and str(p.get("text") or "").strip()
    ]

    allowed_para_ids = [p["para_id"] for p in paras_compact]

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
    anchors_required = int(
        getattr(cfg, "harvest_min_anchors", cfg.anchors_required)
        if harvest_mode else cfg.anchors_required
    )
    anchors_max = 2 if harvest_mode else 4

    payload = {
        "atom": atom_compact,
        "doc_id": doc_id,
        "doc_meta": doc_meta,
        "anchors_required": anchors_required,
        "anchors_max": anchors_max,
        "allowed_para_ids": allowed_para_ids,
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

    return (
        "SYSTEM / OUTPUT CONTRACT (NON-NEGOTIABLE):\n"
        "1) Output ONE JSON object ONLY matching the schema.\n"
        "2) Use ONLY allowed keys.\n"
        "3) Do NOT echo input JSON or evidence paragraphs.\n"
        "4) You MUST choose anchor.para_id ONLY from Input JSON allowed_para_ids.\n"
        f"5) If you cannot meet anchors_required={anchors_required}, "
        "set relevant=false, anchors=[], precedent_score<=40, use_mode=\"contrast\".\n"
        f"6) If relevant=true, return 1..{anchors_max} anchors (never 0).\n\n"
        "You are a legal precedent verifier.\n\n"
        f"{_VERDICT_SCHEMA_TEXT}\n\n"
        "OUTPUT JSON TEMPLATE (fill values, keep keys identical):\n"
        f"{json.dumps(verdict_template, ensure_ascii=False)}\n\n"
        "Input JSON (DO NOT ECHO BACK):\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
    )
