from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests

from ..schemas.verdict import Verdict, VerdictValidator


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_FIRST_OBJ_RE = re.compile(r"\{", re.DOTALL)
_LAST_OBJ_RE = re.compile(r"\}", re.DOTALL)


@dataclass(frozen=True)
class LLMClientConfig:
    model: str = "mistral-small3.2:latest"
    ollama_url: str = "http://localhost:11434/api/generate"
    timeout_s: int = 180
    temperature: float = 0.0
    num_predict: int = 600

    # retry policy
    max_retries: int = 2


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Robust JSON object extraction from model output.
    Accepts raw JSON or JSON surrounded by fences or extra text.
    """
    if not text or not text.strip():
        raise ValueError("Empty model output")

    s = text.strip()
    # If it's pure JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Strip fences
    s2 = _JSON_FENCE_RE.sub("", s).strip()

    # Try again
    try:
        obj = json.loads(s2)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find first {...} span
    start = s2.find("{")
    end = s2.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate JSON object braces in model output")

    candidate = s2[start : end + 1].strip()
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Extracted JSON is not an object")
    return obj


def _ollama_generate(prompt: str, cfg: LLMClientConfig) -> str:
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": cfg.temperature,
            "num_predict": cfg.num_predict,
        },
    }
    r = requests.post(cfg.ollama_url, json=payload, timeout=cfg.timeout_s)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def _ollama_generate(prompt: str, cfg: LLMClientConfig, force_json: bool = True) -> str:
    payload = {
        "model": cfg.model,
        "prompt": prompt,
        "stream": False,
        # ✅ best-effort JSON enforcement (supported by many Ollama models)
        **({"format": "json"} if force_json else {}),
        "options": {
            "temperature": cfg.temperature,
            "num_predict": cfg.num_predict,
        },
    }
    r = requests.post(cfg.ollama_url, json=payload, timeout=cfg.timeout_s)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def verify_with_ollama(prompt: str, cfg: LLMClientConfig) -> Verdict:
    """
    Calls Ollama and returns a validated Verdict.
    Strategy:
      1) Try strict JSON (format=json) + extract/validate.
      2) If it fails, do ONE repair call: "convert your last answer into strict JSON schema".
      3) Retry budget applies to step (1); repair is invoked only after a failure.
    """
    last_err: Optional[Exception] = None
    last_txt: str = ""

    # 1) normal attempts (strict json transport)
    for attempt in range(cfg.max_retries + 1):
        try:
            txt = _ollama_generate(prompt, cfg, force_json=True)
            last_txt = txt

            # If Ollama returns valid JSON object, json.loads will succeed.
            # Still keep robust extraction in case model returns wrapped content.
            obj = _extract_json_object(txt)

            VerdictValidator.validate(obj)
            return Verdict.from_dict(obj)

        except Exception as e:
            last_err = e

            # 2) Repair call (single, decisive)
            try:
                repair_prompt = (
                    "You MUST output a SINGLE JSON object matching the required schema.\n"
                    "NO prose. NO markdown. NO extra keys.\n"
                    "If you cannot find verbatim anchors in the provided paras, set relevant=false and anchors=[].\n\n"
                    "Now convert the following content into the required JSON schema.\n\n"
                    "CONTENT:\n"
                    f"{last_txt}\n\n"
                    "REQUIRED OUTPUT: JSON ONLY.\n"
                )
                txt2 = _ollama_generate(repair_prompt + prompt, cfg, force_json=True)
                obj2 = _extract_json_object(txt2)
                VerdictValidator.validate(obj2)
                return Verdict.from_dict(obj2)

            except Exception:
                # If repair failed, continue retry loop (next attempt)
                prompt = (
                    "IMPORTANT: Your previous output was invalid.\n"
                    "Return STRICT JSON ONLY matching the schema. No extra text.\n\n"
                    + prompt
                )
                continue

    raise RuntimeError(
        f"verify_with_ollama failed after {cfg.max_retries + 1} attempts. "
        f"Last error: {last_err}. Last output (truncated): {last_txt[:1200]!r}"
    )
