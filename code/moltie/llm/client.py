from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from ..schemas.verdict import Verdict, VerdictValidator

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

# If the model starts dumping paragraphs or adding illegal keys, we hard-fail to trigger repair.
# NOTE: keep these specific to illegal TOP-LEVEL keys, not generic substrings.
_ILLEGAL_KEY_SNIPS = (
    '"matched_paras"',
    '"relevant_paras"',
    '"supporting_paragraphs"',
    '"retrieval_explanation"',
    '"retrieval_k"',
    '"retrieval_min_hits"',
    '"retrieval_matched_paras"',
    '"title"',
    '"paras"',  # top-level paras dump indicator
)

_MAX_RAW_OUTPUT_CHARS = 40_000


@dataclass(frozen=True)
class LLMClientConfig:
    model: str = "mistral-small3.2:latest"
    ollama_url: str = "http://localhost:11434/api/generate"
    timeout_s: int = 180
    temperature: float = 0.0
    num_predict: int = 450  # smaller helps reduce rambly JSON breakage
    max_retries: int = 2


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Empty model output")

    s = text.strip()
    s = _JSON_FENCE_RE.sub("", s).strip()

    # Try direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find first {...} span
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate JSON object braces in model output")

    candidate = s[start : end + 1].strip()
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Extracted JSON is not an object")
    return obj


def _ollama_generate(prompt: str, cfg: LLMClientConfig, force_json: bool = True) -> str:
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": cfg.temperature,
            "num_predict": cfg.num_predict,
        },
    }
    if force_json:
        payload["format"] = "json"

    r = requests.post(cfg.ollama_url, json=payload, timeout=cfg.timeout_s)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def _make_repair_prompt(bad_output: str) -> str:
    return (
        "You MUST output ONE SINGLE VALID JSON object.\n"
        "NO prose. NO markdown. NO extra keys.\n"
        "Do not include any paragraph dumps.\n"
        "If you cannot find verbatim anchors in the provided evidence, set relevant=false and anchors=[].\n"
        "precedent_score and confidence MUST be integers 0..100 (never -1). If unknown, use 0.\n\n"
        "REQUIRED VERDICT SCHEMA (keys must match exactly):\n"
        "{"
        '"atom_id":"string","doc_id":"string","relevant":true|false,"matched_X":["X1","X2","X3","X4","X5"],'
        '"precedent_score":0-100,"confidence":0-100,'
        '"anchors":[{"para_id":"p00001","quote":"verbatim substring","why_it_matters":"short"}],'
        '"use_mode":"support|contrast|harmful","proposition_winner":"claimant|respondent|mixed|unclear",'
        '"appeal_outcome":"allowed|dismissed|remitted|mixed|unknown","successful_party":"claimant|respondent|mixed|unclear",'
        '"distinguishers":["..."],"note":"ONE sentence","retrieval_score":null,"retrieval_method":"string|null"'
        "}\n\n"
        "BAD OUTPUT TO FIX (convert it to the schema above):\n"
        + (bad_output or "EMPTY_OUTPUT")
    )


def _extract_from_prompt(prompt_txt: str, key: str) -> Optional[str]:
    """
    Best-effort extraction of a string value from the original base prompt.
    Looks for: "key": "VALUE"
    """
    needle = f"\"{key}\":"
    j = prompt_txt.find(needle)
    if j == -1:
        return None
    k = prompt_txt.find(":", j)
    if k == -1:
        return None
    tail = prompt_txt[k + 1 :].lstrip()
    if not tail.startswith("\""):
        return None
    tail = tail[1:]
    end = tail.find("\"")
    if end == -1:
        return None
    val = tail[:end]
    return val.strip() or None


def _coerce_missing_ids(obj: Dict[str, Any], base_prompt: str) -> Dict[str, Any]:
    if not str(obj.get("atom_id") or "").strip():
        aid = _extract_from_prompt(base_prompt, "atom_id")
        if aid:
            obj["atom_id"] = aid
    if not str(obj.get("doc_id") or "").strip():
        did = _extract_from_prompt(base_prompt, "doc_id")
        if did:
            obj["doc_id"] = did
    return obj


def _clamp_int_0_100(x: Any, default: int = 0) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    if v < 0:
        return 0
    if v > 100:
        return 100
    return v


def _sanitize_verdict_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce hard numeric ranges + cross-field semantics to prevent sentinel poisoning.
    """
    obj["precedent_score"] = _clamp_int_0_100(obj.get("precedent_score"), default=0)
    obj["confidence"] = _clamp_int_0_100(obj.get("confidence"), default=0)

    # relevant must be bool-ish
    rel = obj.get("relevant")
    if isinstance(rel, bool):
        pass
    elif isinstance(rel, (int, float)):
        obj["relevant"] = bool(rel)
    elif isinstance(rel, str):
        obj["relevant"] = rel.strip().lower() in ("true", "1", "yes")
    else:
        obj["relevant"] = False

    # anchors must be list
    if not isinstance(obj.get("anchors"), list):
        obj["anchors"] = []

    # enforce hard rules when relevant is false
    if obj["relevant"] is False:
        obj["use_mode"] = "contrast"
        obj["anchors"] = []
        if obj["precedent_score"] > 40:
            obj["precedent_score"] = 40

    # ensure note is a short sentence fallback (helps downstream)
    note = str(obj.get("note") or "").strip()
    if not note:
        obj["note"] = "No relevant information found."

    return obj


def verify_with_ollama(prompt: str, cfg: LLMClientConfig) -> Verdict:
    """
    Calls Ollama and returns a validated Verdict.

    Strategy:
      A) strict JSON transport (format=json) -> parse -> sanitize -> validate -> return
      B) on failure, ONE repair call: re-emit schema-valid JSON (no full prompt resend)
    """
    base_prompt = prompt  # keep immutable for ID extraction
    last_err: Optional[Exception] = None
    last_txt: str = ""

    # exactly allowed keys at root (your schema says: allowed keys ONLY)
    _ALLOWED_KEYS = {
        "atom_id",
        "doc_id",
        "relevant",
        "matched_X",
        "precedent_score",
        "confidence",
        "anchors",
        "use_mode",
        "proposition_winner",
        "appeal_outcome",
        "successful_party",
        "distinguishers",
        "note",
        "retrieval_score",
        "retrieval_method",
    }

    _WRONG_OBJECT_KEYS = {
        "supporting_paragraphs",
        "retrieval_k",
        "retrieval_min_hits",
        "retrieval_matched_paras",
        "retrieval_explanation",
        "input",
        "paras",
        "matched_paras",
    }

    def _is_wrong_object(obj: Any) -> bool:
        if not isinstance(obj, dict):
            return True
        keys = set(obj.keys())
        if keys & _WRONG_OBJECT_KEYS:
            return True
        # must include core keys
        core = {"atom_id", "doc_id", "relevant", "anchors", "precedent_score", "confidence"}
        return not core.issubset(keys)

    def _enforce_allowed_keys(obj: Dict[str, Any]) -> None:
        keys = set(obj.keys())
        extra = keys - _ALLOWED_KEYS
        missing = _ALLOWED_KEYS - keys
        if extra:
            raise ValueError(f"Verdict JSON included illegal keys: {sorted(list(extra))[:12]}")
        if missing:
            raise ValueError(f"Verdict JSON missing required keys: {sorted(list(missing))[:12]}")

    def _prepend_correction(p: str, why: str) -> str:
        return (
            "IMPORTANT (NON-NEGOTIABLE OUTPUT CONTRACT):\n"
            "- Output ONE JSON object ONLY that matches the Verdict schema.\n"
            "- Use ONLY allowed keys; no extras.\n"
            "- precedent_score and confidence must be integers 0..100 (never -1).\n"
            "- Do NOT output retrieval reports or paragraph dumps.\n"
            f"- Fix required because: {why}\n\n"
            + p
        )

    prompt_to_send = base_prompt

    for attempt in range(cfg.max_retries + 1):
        # ---- Attempt A: normal strict call ----
        try:
            txt = _ollama_generate(prompt_to_send, cfg, force_json=True)
            last_txt = txt

            if len(txt) > _MAX_RAW_OUTPUT_CHARS:
                raise ValueError("Model output too long (likely paragraph dump).")

            if any(sn in txt for sn in _ILLEGAL_KEY_SNIPS):
                raise ValueError("Model output included illegal keys/paragraph dumps.")

            obj = _extract_json_object(txt)
            obj = _coerce_missing_ids(obj, base_prompt)

            if _is_wrong_object(obj):
                raise ValueError(f"Wrong JSON object returned (not a Verdict). Keys={sorted(list(obj.keys()))[:12]}")

            _enforce_allowed_keys(obj)

            obj = _sanitize_verdict_object(obj)

            VerdictValidator.validate(obj)
            return Verdict.from_dict(obj)

        except Exception as e:
            last_err = e
            prompt_to_send = _prepend_correction(base_prompt, why=str(e))

        # ---- Attempt B: repair ----
        try:
            repair_prompt = (
                "REPAIR TASK:\n"
                "Return ONE Verdict JSON object ONLY with EXACT allowed keys.\n"
                "precedent_score and confidence MUST be integers 0..100 (never -1). If unknown, use 0.\n"
                "Do NOT output retrieval reports or paragraph dumps.\n\n"
                + _make_repair_prompt(last_txt)
            )

            txt2 = _ollama_generate(repair_prompt, cfg, force_json=True)

            if len(txt2) > _MAX_RAW_OUTPUT_CHARS:
                raise ValueError("Repair output too long (likely paragraph dump).")

            obj2 = _extract_json_object(txt2)
            obj2 = _coerce_missing_ids(obj2, base_prompt)

            if _is_wrong_object(obj2):
                raise ValueError(f"Repair returned wrong JSON object. Keys={sorted(list(obj2.keys()))[:12]}")

            _enforce_allowed_keys(obj2)

            obj2 = _sanitize_verdict_object(obj2)

            VerdictValidator.validate(obj2)
            return Verdict.from_dict(obj2)

        except Exception as e2:
            last_err = e2
            prompt_to_send = _prepend_correction(base_prompt, why=str(e2))
            continue

    raise RuntimeError(
        f"verify_with_ollama failed after {cfg.max_retries + 1} attempts. "
        f"Last error: {last_err}. Last output (truncated): {last_txt[:1200]!r}"
    )
