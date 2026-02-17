from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, List  # ✅ FIX: import List

import requests

from ..schemas.verdict import Verdict, VerdictValidator


"""
moltie.llm.client
=================

Strict LLM client for generating and validating Verdict objects.

Purpose
-------
This module wraps calls to a local Ollama model and enforces a deterministic
output contract. It ensures that all LLM responses conform exactly to the
Verdict schema used by the Moltie reasoning engine.

The client converts probabilistic model output into a validated, typed
Verdict domain object. If the model output violates the contract, the client
attempts controlled repair before ultimately failing loudly.

Core Responsibilities
---------------------

1. Transport
   - Calls Ollama using format="json".
   - Controls temperature and token limits.

2. JSON Extraction & Salvage
   - Extracts the first JSON object from model output.
   - Repairs common JSON breakage (e.g. raw newlines inside strings).
   - Handles truncated anchors arrays safely.

3. Schema Enforcement
   - Drops illegal root keys.
   - Ensures all required Verdict keys exist.
   - Normalizes enums and value ranges.
   - Coerces types (ints, bools, lists, strings).
   - Maps common synonym values (e.g. employer → respondent).

4. Anchor Validation
   - Requires anchors to be objects with:
        {para_id, quote, why_it_matters}
   - Accepts "text" as alias for "quote".
   - Removes malformed anchors.
   - Enforces low-signal rule:
        No anchors → relevant=False and score capped.

5. Contract Repair
   - On invalid output, sends a strict repair prompt.
   - Limits retries.
   - Raises RuntimeError if the contract cannot be satisfied.

Design Philosophy
-----------------
This client acts as a boundary guard between the LLM and the reasoning engine.
It ensures:

    LLM output → Validated Verdict → Safe downstream execution

The rest of Moltie assumes Verdict objects are structurally correct.
This module guarantees that invariant.

Failure Modes
-------------
- Invalid JSON → salvage attempt → repair attempt → RuntimeError
- Schema violations → repair attempt → RuntimeError
- Missing anchors → forced relevant=False

This is intentional. Silent corruption is worse than failure.

Authoritative Output
--------------------
The only acceptable final output from verify_with_ollama() is:

    Verdict

All other states are rejected.
"""

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_VERDICT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "atom_id": {"type": "string"},
        "doc_id": {"type": "string"},
        "relevant": {"type": "boolean"},
        "matched_X": {"type": "array", "items": {"type": "string"}},
        "precedent_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "anchors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "para_id": {"type": "string"},
                    "quote": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                },
                "required": ["para_id", "quote", "why_it_matters"],
                "additionalProperties": False,
            },
        },
        "use_mode": {"type": "string"},
        "proposition_winner": {"type": "string"},
        "appeal_outcome": {"type": "string"},
        "successful_party": {"type": "string"},
        "distinguishers": {"type": "array", "items": {"type": "string"}},
        "note": {"type": "string"},
        "retrieval_score": {"type": ["number", "null"]},
        "retrieval_method": {"type": "string"},
    },
    "required": [
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
    ],
    "additionalProperties": False,
}

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
    num_predict: int = 1200  # smaller helps reduce rambly JSON breakage
    max_retries: int = 2


def _escape_raw_newlines_in_json_strings(raw: str) -> str:
    """
    JSON forbids literal newlines inside strings. Some models emit them.
    This converts raw newline characters that occur *inside* JSON strings into \\n / \\r.

    IMPORTANT: does NOT collapse whitespace outside strings (so it won't destroy structure).
    """
    out: List[str] = []
    in_str = False
    esc = False

    for ch in raw:
        if in_str:
            if esc:
                esc = False
                out.append(ch)
                continue
            if ch == "\\":
                esc = True
                out.append(ch)
                continue
            if ch == "\"":
                in_str = False
                out.append(ch)
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            out.append(ch)
        else:
            if ch == "\"":
                in_str = True
            out.append(ch)

    return "".join(out)


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Empty model output")

    s = text.strip()
    s = _JSON_FENCE_RE.sub("", s).strip()

    # 1) Direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) Extract first {...} blob
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate JSON object braces in model output")

    candidate = s[start : end + 1].strip()

    # Helper: replace a JSON array field value with [] using a quote-aware scan
    def _replace_array_field_with_empty(raw: str, key: str) -> Optional[str]:
        needle = f"\"{key}\""
        i = raw.find(needle)
        if i == -1:
            return None

        # Find ':' after the key
        j = raw.find(":", i + len(needle))
        if j == -1:
            return None

        # Find '[' that begins the array (skip whitespace)
        k = j + 1
        n = len(raw)
        while k < n and raw[k].isspace():
            k += 1
        if k >= n or raw[k] != "[":
            return None

        # Scan forward to find the matching closing ']' (quote-aware)
        in_str = False
        esc = False
        depth = 0
        m = k
        while m < n:
            ch = raw[m]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == "\"":
                    in_str = False
            else:
                if ch == "\"":
                    in_str = True
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        # replace [ ... ] with []
                        return raw[:k] + "[]" + raw[m + 1 :]
            m += 1

        # FALLBACK: unterminated array (typical truncation). We still salvage by:
        # - replacing from '[' to the final root '}' with []
        # Candidate raw should already include the closing root brace (because you slice start..rfind("}")).
        root_end = raw.rfind("}")
        if root_end == -1 or root_end <= k:
            return None

        # Keep everything up to '[' then inject [], then close object immediately.
        # This intentionally drops any partial garbage after the unterminated anchors.
        return raw[:k] + "[]\n" + raw[root_end:]

    # 3) Parse candidate; salvage if needed WITHOUT collapsing whitespace globally
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e0:
        # 3a) Escape illegal raw newlines inside strings
        candidate_fixed = _escape_raw_newlines_in_json_strings(candidate)

        try:
            obj = json.loads(candidate_fixed)
        except json.JSONDecodeError as e2:
            # 3b) If still broken, strip anchors array and retry parse
            repaired = _replace_array_field_with_empty(candidate_fixed, "anchors")
            if repaired:
                try:
                    obj = json.loads(repaired)
                except json.JSONDecodeError as e3:
                    pos = getattr(e3, "pos", None)
                    if isinstance(pos, int):
                        lo = max(0, pos - 140)
                        hi = min(len(repaired), pos + 140)
                        snippet = repaired[lo:hi]
                        raise ValueError(
                            f"JSON salvage failed even after stripping anchors at pos={pos}. Nearby snippet: {snippet!r}"
                        ) from e3
                    raise ValueError(f"JSON salvage failed even after stripping anchors: {e3}") from e3
            else:
                # Deterministic debug: show failure position + local context
                pos = getattr(e2, "pos", None)
                if isinstance(pos, int):
                    lo = max(0, pos - 140)
                    hi = min(len(candidate_fixed), pos + 140)
                    snippet = candidate_fixed[lo:hi]
                    raise ValueError(
                        f"JSON salvage failed at pos={pos}. Nearby snippet: {snippet!r}"
                    ) from e2
                raise ValueError(f"JSON salvage failed: {e2}") from e2

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
        payload["format"] = _VERDICT_JSON_SCHEMA

    r = requests.post(cfg.ollama_url, json=payload, timeout=cfg.timeout_s)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def _make_repair_prompt(bad_output: str) -> str:
    return (
        "You MUST output ONE SINGLE VALID JSON object.\n"
        "NO prose. NO markdown. NO extra keys.\n"
        "Do not include any paragraph dumps.\n"
        "IMPORTANT: anchors MUST be a list of objects (not strings). If you cannot provide para_id, set anchors=[].\n"
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
    """
    Fill atom_id/doc_id from the Input JSON payload in the prompt.

    Treat placeholders (e.g. "", "string", all-zero UUID, "none", "unknown")
    as invalid and overwrite them.

    atom_id/doc_id are metadata, not model judgment, so overriding bad values
    is correct and safe.
    """
    if not isinstance(obj, dict):
        return obj

    _BAD_VALUES = {
        "",
        "string",
        "none",
        "null",
        "unknown",
        "00000000-0000-0000-0000-000000000000",
    }

    def _bad(v: Any) -> bool:
        s = ("" if v is None else str(v)).strip().lower()
        return s in _BAD_VALUES

    # Extract the Input JSON block (payload) from prompt
    payload = None
    try:
        marker = "Input JSON"
        i = base_prompt.rfind(marker)
        if i != -1:
            tail = base_prompt[i:]
            payload = _extract_json_object(tail)
    except Exception:
        payload = None

    if not isinstance(payload, dict):
        return obj

    # Pull authoritative IDs from payload
    true_doc_id = str(payload.get("doc_id") or "").strip()

    true_atom_id = ""
    atom = payload.get("atom")
    if isinstance(atom, dict):
        true_atom_id = str(atom.get("atom_id") or "").strip()

    # Overwrite if missing OR placeholder
    if true_doc_id and (_bad(obj.get("doc_id")) or obj.get("doc_id") != true_doc_id):
        obj["doc_id"] = true_doc_id

    if true_atom_id and (_bad(obj.get("atom_id")) or obj.get("atom_id") != true_atom_id):
        obj["atom_id"] = true_atom_id

    return obj

def _sanitize_verdict_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Defensive normalizer for Verdict JSON objects before validation.

    Goals:
    - Drop unknown root keys (so we can sanitize before enforcing allowed keys).
    - Ensure ALL required keys exist (fill defaults).
    - Coerce types (ints, bools, lists, strings).
    - Normalize enums (use_mode, outcomes, winners).
    - Accept common model variants:
        * anchors items may use "text" instead of "quote"
        * winners may be "employer"/"employee" -> respondent/claimant
    - Ensure anchors are well-formed dicts with required fields (para_id, quote, why_it_matters).
    - Enforce low-signal rule: no anchors => relevant=false and cap score.
    """

    # ========= contract keys (root) =========
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

    # Drop illegal/unknown root keys (e.g., "precedent_type", "why_it_matters", etc.)
    if isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if k in _ALLOWED_KEYS}
    else:
        obj = {}

    # Ensure required keys exist (defaults)
    defaults: Dict[str, Any] = {
        "atom_id": "",
        "doc_id": "",
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
        "note": "No relevant information found.",
        "retrieval_score": None,
        "retrieval_method": None,
    }
    for k, v in defaults.items():
        obj.setdefault(k, v)

    # ---------- helpers ----------
    def _as_str(x: Any) -> str:
        return str(x) if x is not None else ""

    def _clean_str(x: Any) -> str:
        return _as_str(x).strip()

    def _clamp_int(x: Any, lo: int = 0, hi: int = 100, default: int = 0) -> int:
        try:
            v = int(x)
        except Exception:
            return default
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    def _as_bool(x: Any) -> bool:
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {"true", "t", "yes", "y", "1"}:
                return True
            if s in {"false", "f", "no", "n", "0", ""}:
                return False
        return False

    def _map_party(x: Any) -> str:
        s = _clean_str(x).lower()
        # common synonyms
        if s in {"claimant", "employee", "appellant", "worker"}:
            return "claimant"
        if s in {"respondent", "employer", "company"}:
            return "respondent"
        if s == "mixed":
            return "mixed"
        return "unclear"

    # ---------- core fields ----------
    obj["atom_id"] = _clean_str(obj.get("atom_id"))
    obj["doc_id"] = _clean_str(obj.get("doc_id"))

    obj["relevant"] = _as_bool(obj.get("relevant", False))

    obj["precedent_score"] = _clamp_int(obj.get("precedent_score", 0), 0, 100, 0)
    obj["confidence"] = _clamp_int(obj.get("confidence", 0), 0, 100, 0)

    # matched_X: list[str] but ONLY X1..X5
    mx = obj.get("matched_X", [])
    if not isinstance(mx, list):
        mx = []

    allowed_mx = {"X1", "X2", "X3", "X4", "X5"}
    mx2: List[str] = []
    for x in mx:
        s = _clean_str(x).upper()
        if s in allowed_mx:
            mx2.append(s)

    # de-dup preserve order
    seen = set()
    mx3: List[str] = []
    for s in mx2:
        if s not in seen:
            seen.add(s)
            mx3.append(s)

    obj["matched_X"] = mx3

    # distinguishers: list[str]
    dist = obj.get("distinguishers", [])
    if not isinstance(dist, list):
        dist = []
    dist2: List[str] = []
    for x in dist:
        s = _clean_str(x)
        if s:
            dist2.append(s)
    obj["distinguishers"] = dist2

    # note: keep non-empty
    note = _clean_str(obj.get("note", ""))
    obj["note"] = note if note else "No relevant information found."

    # retrieval fields
    rm = obj.get("retrieval_method", None)
    obj["retrieval_method"] = None if rm is None else (_clean_str(rm) or None)

    rs = obj.get("retrieval_score", None)
    if rs is None:
        obj["retrieval_score"] = None
    else:
        try:
            obj["retrieval_score"] = float(rs)
        except Exception:
            obj["retrieval_score"] = None

    # ---------- anchors ----------
    anchors = obj.get("anchors", [])
    if not isinstance(anchors, list):
        anchors = []

    anchors_out: List[Dict[str, str]] = []
    for a in anchors:
        if not isinstance(a, dict):
            continue

        pid = _clean_str(a.get("para_id"))

        # Accept either "quote" or "text" (model often emits "text")
        q_raw = a.get("quote", None)
        if q_raw is None:
            q_raw = a.get("text", None)
        quote = _as_str(q_raw).strip()

        why = _clean_str(a.get("why_it_matters")) or "Relevant passage."

        if not pid or not quote:
            continue

        anchors_out.append({"para_id": pid, "quote": quote, "why_it_matters": why})

    obj["anchors"] = anchors_out

    # ---------- enum normalization ----------
    # ---------- enum normalization ----------
    rel = bool(obj.get("relevant", False))

    um = _clean_str(obj.get("use_mode", "")).lower()
    if um not in {"support", "contrast", "harmful"}:
        um = "support" if rel else "contrast"
    if not rel:
        um = "contrast"
    obj["use_mode"] = um

    obj["proposition_winner"] = _map_party(obj.get("proposition_winner", ""))
    obj["successful_party"] = _map_party(obj.get("successful_party", ""))

    ao = _clean_str(obj.get("appeal_outcome", "")).lower()
    if ao not in {"allowed", "dismissed", "remitted", "mixed", "unknown"}:
        ao = "unknown"
    obj["appeal_outcome"] = ao

    # ---------- low-signal rule ----------
    if len(obj["anchors"]) == 0:
        obj["relevant"] = False
        obj["use_mode"] = "contrast"
        obj["precedent_score"] = min(obj["precedent_score"], 40)

    if not obj["relevant"]:
        obj["precedent_score"] = min(obj["precedent_score"], 40)

    # ✅ FINAL CONSISTENCY GUARD
    # If relevant=True, "contrast" makes no sense for your semantics.
    if obj["relevant"] and obj["use_mode"] == "contrast":
        obj["use_mode"] = "support"

    return obj


def verify_with_ollama(prompt: str, cfg: LLMClientConfig) -> Verdict:
    """
    Calls Ollama and returns a validated Verdict.

    ADD-ONLY DEBUG VERSION:
    - Prints prompt hash/head
    - Prints raw model output head/tail
    - Prints extracted keys + atom_id/doc_id before/after coercion
    - Prints exception causes for Attempt A and Attempt B
    """
    base_prompt = prompt
    last_err: Optional[Exception] = None
    last_txt: str = ""

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

    # --- ADDED: local hash helper (client.py may not have loop._h) ---
    def _h(s: str) -> str:
        import hashlib
        return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()[:10]

    def _is_wrong_object(obj: Any) -> bool:
        if not isinstance(obj, dict):
            return True
        keys = set(obj.keys())
        if keys & _WRONG_OBJECT_KEYS:
            return True
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
            "- anchors MUST be a list of objects, never strings. If you cannot provide para_id, output anchors=[].\n"
            "- Do NOT output retrieval reports or paragraph dumps.\n"
            "- Do NOT output an 'answer' field or any wrapper object. Output the Verdict object ONLY.\n"
            "- NEVER output a JSON object with a single key like {'answer': ...}. That is invalid.\n"
            "- You must output ALL required keys, including anchors/use_mode/etc.\n"
            f"- Fix required because: {why}\n\n"
            + p
        )

    prompt_to_send = base_prompt

    for attempt in range(cfg.max_retries + 1):
        # ---- Attempt A ----
        try:
            txt = _ollama_generate(prompt_to_send, cfg, force_json=True)
            last_txt = txt

            # --- ADDED DEBUG (Attempt A) ---
            try:
                print("\n[moltie.client] ===== Attempt A =====")
                print("[moltie.client] attempt:", attempt)
                print("[moltie.client] prompt_hash:", _h(prompt_to_send))
                print("[moltie.client] prompt_head:\n", (prompt_to_send or "")[:600])
                print("[moltie.client] raw_len:", len(txt or ""))
                print("[moltie.client] raw_head:\n", (txt or "")[:800])
                print("[moltie.client] raw_tail:\n", (txt or "")[-400:])
            except Exception as _dbg_e:
                print("[moltie.client] debug_print_failed:", repr(_dbg_e))

            if len(txt) > _MAX_RAW_OUTPUT_CHARS:
                raise ValueError("Model output too long (likely paragraph dump).")

            # NOTE: do NOT substring-scan for illegal keys here; it can false-trigger inside quotes.
            obj = _extract_json_object(txt)

            # --- ADDED DEBUG: extracted obj keys (Attempt A) ---
            try:
                print("[moltie.client] extracted_keys:", sorted(list((obj or {}).keys()))[:40])
                print("[moltie.client] extracted_atom_id:", (obj or {}).get("atom_id"))
                print("[moltie.client] extracted_doc_id:", (obj or {}).get("doc_id"))
            except Exception as _dbg_e:
                print("[moltie.client] debug_obj_failed:", repr(_dbg_e))

            obj = _coerce_missing_ids(obj, base_prompt)

            # --- ADDED DEBUG: post-coerce ids (Attempt A) ---
            try:
                print("[moltie.client] post_coerce_atom_id:", (obj or {}).get("atom_id"))
                print("[moltie.client] post_coerce_doc_id:", (obj or {}).get("doc_id"))
            except Exception as _dbg_e:
                print("[moltie.client] debug_post_coerce_failed:", repr(_dbg_e))

            if _is_wrong_object(obj):
                raise ValueError(f"Wrong JSON object returned (not a Verdict). Keys={sorted(list(obj.keys()))[:12]}")

            _enforce_allowed_keys(obj)

            obj = _sanitize_verdict_object(obj)

            VerdictValidator.validate(obj)
            return Verdict.from_dict(obj)

        except Exception as e:
            # --- ADDED DEBUG: why Attempt A failed ---
            try:
                print("[moltie.client] Attempt A exception:", repr(e))
            except Exception:
                pass

            last_err = e
            prompt_to_send = _prepend_correction(base_prompt, why=str(e))

        # ---- Attempt B: repair ----
        try:
            repair_prompt = (
                "REPAIR TASK:\n"
                "Return ONE Verdict JSON object ONLY with EXACT allowed keys.\n"
                "precedent_score and confidence MUST be integers 0..100 (never -1). If unknown, use 0.\n"
                "anchors MUST be a list of objects, never strings. If you cannot provide para_id, set anchors=[].\n"
                "Do NOT output retrieval reports or paragraph dumps.\n\n"
                "If anchors is empty, set relevant=false and use_mode='contrast' and precedent_score<=40.\n"
                + "You MUST use the following Input JSON evidence. Copy quotes EXACTLY as substrings.\n\n"
                + base_prompt
                + "\n\nMODEL LAST OUTPUT (for reference only; do not copy text unless it appears verbatim in evidence):\n"
                + _make_repair_prompt(last_txt)
            )

            txt2 = _ollama_generate(repair_prompt, cfg, force_json=True)

            # --- ADDED DEBUG (Attempt B) ---
            try:
                print("\n[moltie.client] ===== Attempt B (REPAIR) =====")
                print("[moltie.client] attempt:", attempt)
                print("[moltie.client] repair_prompt_hash:", _h(repair_prompt))
                print("[moltie.client] repair_prompt_head:\n", (repair_prompt or "")[:600])
                print("[moltie.client] raw2_len:", len(txt2 or ""))
                print("[moltie.client] raw2_head:\n", (txt2 or "")[:800])
                print("[moltie.client] raw2_tail:\n", (txt2 or "")[-400:])
            except Exception as _dbg_e:
                print("[moltie.client] debug_print_failed:", repr(_dbg_e))

            if len(txt2) > _MAX_RAW_OUTPUT_CHARS:
                raise ValueError("Repair output too long (likely paragraph dump).")

            obj2 = _extract_json_object(txt2)

            # --- ADDED DEBUG: extracted obj2 keys (Attempt B) ---
            try:
                print("[moltie.client] extracted2_keys:", sorted(list((obj2 or {}).keys()))[:40])
                print("[moltie.client] extracted2_atom_id:", (obj2 or {}).get("atom_id"))
                print("[moltie.client] extracted2_doc_id:", (obj2 or {}).get("doc_id"))
            except Exception as _dbg_e:
                print("[moltie.client] debug_obj2_failed:", repr(_dbg_e))

            obj2 = _coerce_missing_ids(obj2, base_prompt)

            # --- ADDED DEBUG: post-coerce ids (Attempt B) ---
            try:
                print("[moltie.client] post_coerce2_atom_id:", (obj2 or {}).get("atom_id"))
                print("[moltie.client] post_coerce2_doc_id:", (obj2 or {}).get("doc_id"))
            except Exception as _dbg_e:
                print("[moltie.client] debug_post_coerce2_failed:", repr(_dbg_e))

            if _is_wrong_object(obj2):
                raise ValueError(f"Repair returned wrong JSON object. Keys={sorted(list(obj2.keys()))[:12]}")

            _enforce_allowed_keys(obj2)

            obj2 = _sanitize_verdict_object(obj2)

            VerdictValidator.validate(obj2)
            return Verdict.from_dict(obj2)

        except Exception as e2:
            # --- ADDED DEBUG: why Attempt B failed ---
            try:
                print("[moltie.client] Attempt B exception:", repr(e2))
            except Exception:
                pass

            last_err = e2
            prompt_to_send = _prepend_correction(base_prompt, why=str(e2))
            continue

    raise RuntimeError(
        f"verify_with_ollama failed after {cfg.max_retries + 1} attempts. "
        f"Last error: {last_err}. Last output (truncated): {last_txt[:1200]!r}"
    )
