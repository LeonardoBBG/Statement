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
    '"supporting_paragraphs"',
    '"retrieval_explanation"',
    '"retrieval_k"',
    '"retrieval_min_hits"',
    '"retrieval_matched_paras"',
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

    # --- required keys for a Verdict root object ---
    _REQUIRED_KEYS = {
        "atom_id",
        "doc_id",
        "relevant",
        "matched_X",
        "precedent_score",
        "confidence",
        "anchors",
        "use_mode",
        "note",
        "retrieval_score",
        "retrieval_method",
        "proposition_winner",
        "appeal_outcome",
        "successful_party",
        "distinguishers",
    }

    # --- keys that strongly indicate "wrong object" (retrieval report / echo) ---
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

    def _extract_from_prompt(prompt_txt: str, key: str) -> Optional[str]:
        """
        Best-effort extraction of a string value from the prompt without regex.
        Looks for: "key": "VALUE"
        """
        needle = f"\"{key}\":"
        j = prompt_txt.find(needle)
        if j == -1:
            return None
        # move to after colon
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

    def _coerce_missing_ids(obj: Dict[str, Any], prompt_txt: str) -> Dict[str, Any]:
        """
        If model omits atom_id/doc_id, force-fill from prompt (template/input contains them).
        """
        if not str(obj.get("atom_id") or "").strip():
            aid = _extract_from_prompt(prompt_txt, "atom_id")
            if aid:
                obj["atom_id"] = aid
        if not str(obj.get("doc_id") or "").strip():
            did = _extract_from_prompt(prompt_txt, "doc_id")
            if did:
                obj["doc_id"] = did
        return obj

    def _is_wrong_object(obj: Any) -> bool:
        if not isinstance(obj, dict):
            return True
        keys = set(obj.keys())
        if keys & _WRONG_OBJECT_KEYS:
            return True
        # If it doesn't even contain core Verdict keys, it's wrong
        if not {"atom_id", "doc_id", "relevant", "anchors", "precedent_score", "confidence"} <= keys:
            return True
        return False

    def _prepend_correction(p: str, why: str) -> str:
        return (
            "IMPORTANT (NON-NEGOTIABLE OUTPUT CONTRACT):\n"
            "- Output ONE JSON object ONLY that matches the Verdict schema.\n"
            "- Do NOT output retrieval reports. Do NOT output supporting_paragraphs or retrieval_explanation.\n"
            "- Use ONLY allowed keys from the Verdict schema.\n"
            "- You MUST include atom_id and doc_id (copy exactly from the input/template).\n"
            f"- Fix required because: {why}\n\n"
            + p
        )

    for attempt in range(cfg.max_retries + 1):
        # ---- Attempt A: normal strict call ----
        try:
            txt = _ollama_generate(prompt, cfg, force_json=True)
            last_txt = txt

            if len(txt) > _MAX_RAW_OUTPUT_CHARS:
                raise ValueError("Model output too long (likely paragraph dump).")

            if any(sn in txt for sn in _ILLEGAL_KEY_SNIPS):
                raise ValueError("Model output included illegal keys/paragraph dumps.")

            obj = _extract_json_object(txt)

            # force-fill atom_id/doc_id if omitted
            if isinstance(obj, dict):
                obj = _coerce_missing_ids(obj, prompt)

            # reject wrong-object JSON (retrieval report / echo)
            if _is_wrong_object(obj):
                raise ValueError(f"Wrong JSON object returned (not a Verdict). Keys={sorted(list(obj.keys()))[:12] if isinstance(obj, dict) else type(obj)}")

            # required keys check (pre-validation; gives better error message)
            missing = [k for k in _REQUIRED_KEYS if k not in obj]
            if missing:
                raise ValueError(f"Verdict JSON missing required keys: {missing[:8]}")

            VerdictValidator.validate(obj)
            return Verdict.from_dict(obj)

        except Exception as e:
            last_err = e
            prompt = _prepend_correction(prompt, why=str(e))

        # ---- Attempt B: repair ----
        try:
            repair_prompt = _make_repair_prompt(last_txt if last_txt else "EMPTY_OUTPUT")
            repair_prompt = (
                "REPAIR TASK:\n"
                "You previously output the WRONG JSON object.\n"
                "Return ONE Verdict JSON object ONLY with EXACT allowed keys.\n"
                "Do NOT output retrieval reports (no supporting_paragraphs, retrieval_explanation, retrieval_k, etc.).\n"
                "You MUST include atom_id and doc_id exactly as in the input/template.\n"
                "If uncertain, set relevant=false, anchors=[], precedent_score<=40, use_mode=\"contrast\".\n\n"
                + repair_prompt
            )

            txt2 = _ollama_generate(repair_prompt, cfg, force_json=True)

            if len(txt2) > _MAX_RAW_OUTPUT_CHARS:
                raise ValueError("Repair output too long (likely paragraph dump).")

            obj2 = _extract_json_object(txt2)

            if isinstance(obj2, dict):
                obj2 = _coerce_missing_ids(obj2, prompt)

            if _is_wrong_object(obj2):
                raise ValueError(f"Repair returned wrong JSON object. Keys={sorted(list(obj2.keys()))[:12] if isinstance(obj2, dict) else type(obj2)}")

            missing2 = [k for k in _REQUIRED_KEYS if k not in obj2]
            if missing2:
                raise ValueError(f"Repair Verdict missing required keys: {missing2[:8]}")

            VerdictValidator.validate(obj2)
            return Verdict.from_dict(obj2)

        except Exception as e2:
            last_err = e2
            prompt = (
                "IMPORTANT: Output STRICT JSON ONLY matching the Verdict schema. "
                "Use only allowed keys. Do NOT echo input JSON. Do NOT output retrieval reports.\n\n"
                + prompt
            )
            continue

    raise RuntimeError(
        f"verify_with_ollama failed after {cfg.max_retries + 1} attempts. "
        f"Last error: {last_err}. Last output (truncated): {last_txt[:1200]!r}"
    )

