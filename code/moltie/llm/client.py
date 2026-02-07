from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from ..schemas.verdict import Verdict, VerdictValidator

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

# If the model starts dumping paragraphs or adding illegal keys, we hard-fail to trigger repair.
_ILLEGAL_KEY_SNIPS = (
    '"matched_paras"',
    '"relevant_paras"',
    '"title"',
    '"paras"',
    '"text": "',
)

# If output is too large, it tends to be paragraph-dump territory.
_MAX_RAW_OUTPUT_CHARS = 40_000


@dataclass(frozen=True)
class LLMClientConfig:
    model: str = "mistral-small3.2:latest"
    ollama_url: str = "http://localhost:11434/api/generate"
    timeout_s: int = 180
    temperature: float = 0.0
    num_predict: int = 450  # ✅ smaller helps reduce rambly JSON breakage

    # retry policy
    max_retries: int = 2


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Empty model output")

    s = text.strip()

    # Strip fences first (common case)
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

    # ✅ best-effort JSON enforcement (supported by many Ollama models)
    if force_json:
        payload["format"] = "json"

    r = requests.post(cfg.ollama_url, json=payload, timeout=cfg.timeout_s)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def _make_repair_prompt(bad_output: str) -> str:
    """
    Strict “re-emit as JSON” repair. Do NOT include the original long prompt again.
    We only feed the bad output back and demand the exact schema.
    """
    return (
        "You MUST output ONE SINGLE VALID JSON object.\n"
        "NO prose. NO markdown. NO extra keys.\n"
        "Do not include any paragraph dumps. Do not include arrays of paragraphs.\n"
        "If you cannot find verbatim anchors in the provided evidence, set relevant=false and anchors=[].\n\n"
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
        + bad_output
    )


def verify_with_ollama(prompt: str, cfg: LLMClientConfig) -> Verdict:
    """
    Calls Ollama and returns a validated Verdict.

    Strategy:
      A) Try strict JSON transport (format=json) -> parse -> validate -> return
      B) On failure, run ONE repair call that re-emits schema-valid JSON
      C) Retry budget applies to (A); repair is attempted once per failed attempt
    """
    last_err: Optional[Exception] = None
    last_txt: str = ""

    # small helper to avoid repeating the same correction text
    def _prepend_correction(p: str, why: str) -> str:
        return (
            "IMPORTANT (NON-NEGOTIABLE OUTPUT CONTRACT):\n"
            "- Output ONE JSON object ONLY.\n"
            "- Use ONLY allowed keys from the Verdict schema.\n"
            "- Do NOT echo Input JSON. Do NOT output evidence paragraphs.\n"
            "- Do NOT output keys like matched_paras / paras / input / analysis.\n"
            f"- Fix required because: {why}\n\n"
            + p
        )

    for attempt in range(cfg.max_retries + 1):
        # ---- Attempt A: normal strict call ----
        try:
            txt = _ollama_generate(prompt, cfg, force_json=True)
            last_txt = txt

            # hard tripwires to trigger repair (paragraph-dump / invented keys)
            if len(txt) > _MAX_RAW_OUTPUT_CHARS:
                raise ValueError("Model output too long (likely paragraph dump).")

            # NOTE: This catches the specific failure you saw: {"matched_paras":[...]}
            # Keep this as a fast pre-parse reject to force repair.
            if '"matched_paras"' in txt:
                raise ValueError('Model returned "matched_paras" instead of Verdict object.')

            if any(sn in txt for sn in _ILLEGAL_KEY_SNIPS):
                # allow "text" only if it appears inside anchors.quote, but models often dump paras
                # so we hard-fail and let repair produce a minimal verdict
                raise ValueError("Model output included illegal keys/paragraph dumps.")

            obj = _extract_json_object(txt)

            # extra guard: reject wrong-root objects even if valid JSON
            # (e.g., model echoes input payload or returns retrieval object)
            if isinstance(obj, dict) and "matched_paras" in obj:
                raise ValueError('Parsed JSON contains "matched_paras" (wrong object).')

            VerdictValidator.validate(obj)
            return Verdict.from_dict(obj)

        except Exception as e:
            last_err = e
            # tighten prompt *specifically* based on what happened
            prompt = _prepend_correction(prompt, why=str(e))

        # ---- Attempt B: repair ----
        try:
            repair_prompt = _make_repair_prompt(last_txt if last_txt else "EMPTY_OUTPUT")
            # strengthen repair: tell it exactly what went wrong without changing variable names
            repair_prompt = (
                "REPAIR TASK:\n"
                "You previously output the WRONG JSON shape.\n"
                "Return ONE Verdict JSON object ONLY (schema-compliant) with ONLY allowed keys.\n"
                "Do NOT include matched_paras, paras, input, analysis, or any evidence dump.\n"
                "If uncertain, set relevant=false, anchors=[], precedent_score<=40, use_mode=\"contrast\".\n\n"
                + repair_prompt
            )

            txt2 = _ollama_generate(repair_prompt, cfg, force_json=True)

            # Repair output should be small and schema-only
            if len(txt2) > _MAX_RAW_OUTPUT_CHARS:
                raise ValueError("Repair output too long (likely paragraph dump).")

            if '"matched_paras"' in txt2:
                raise ValueError('Repair returned "matched_paras" (still wrong object).')

            obj2 = _extract_json_object(txt2)

            if isinstance(obj2, dict) and "matched_paras" in obj2:
                raise ValueError('Parsed repair JSON contains "matched_paras" (wrong object).')

            VerdictValidator.validate(obj2)
            return Verdict.from_dict(obj2)

        except Exception as e2:
            last_err = e2
            # tighten the next attempt slightly without bloating the whole prompt
            prompt = (
                "IMPORTANT: Output STRICT JSON ONLY matching the Verdict schema. "
                "Use only allowed keys. No paragraph dumps. Do NOT echo input JSON.\n\n"
                + prompt
            )
            continue

    raise RuntimeError(
        f"verify_with_ollama failed after {cfg.max_retries + 1} attempts. "
        f"Last error: {last_err}. Last output (truncated): {last_txt[:1200]!r}"
    )
