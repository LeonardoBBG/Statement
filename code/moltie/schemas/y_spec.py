#!/usr/bin/env python3
"""
y_spec.py

Generate an inferred rubric Y (Y_inferred_v2) from Witness Statement (WS) text using Ollama.

Upgrades vs v1:
- Tests are PATTERN-based, scoped, and non-vague:
  Each Xi has: scope, pattern (IF...THEN), required_elements (3..6)
- Deterministic quality gates (reject vague tests)
- Merge prefers specificity + orthogonality (penalize overlap)
- Robust JSON extraction + one retry
- Debug dumps of raw Ollama outputs on failure

Usage:
  python y_spec.py --ws-file ./ws.txt --out ./Y_inferred_v2.json
  cat ws.txt | python y_spec.py --out ./Y_inferred_v2.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# =========================
# Debug dump config
# =========================
_DEBUG_DIR = Path("/home/hello/Projects/Statements/output/debug_y")
_DEBUG_DIR.mkdir(parents=True, exist_ok=True)

def _dump_raw(tag: str, txt: str) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    p = _DEBUG_DIR / f"{ts}_{tag}.txt"
    p.write_text(txt or "", encoding="utf-8")


# =========================
# PROMPT: INFER Y (V2)
# =========================
Y_PROMPT_INFER_V2 = """
You are to INFER an evaluation rubric Y from the fact pattern X.

OUTPUT:
Return VALID JSON ONLY (no markdown, no code fences, no commentary).
The output must start with { and end with }.
End after the final closing brace } and output NOTHING else.

GOAL:
Create 4 to 7 ATOMIC tests that capture the core factual/legal signals in X, such that a short case description
("reasoning_for_index" or "summary") can be classified for usefulness.

CRITICAL: Each test must be NON-VAGUE and PATTERN-BASED.
- Every Xi MUST specify what process it refers to via a scope label.
- Every Xi MUST include a falsifiable pattern and 3..6 required elements (conjunctive conditions).
- Avoid umbrella concepts like "procedural integrity" unless you define it as concrete failures.

ALLOWED scope values (use EXACTLY one per test):
- "INTERNAL_DISCIPLINARY_APPEAL"
- "GRIEVANCE"
- "ET"
- "EAT"
- "GENERAL"   (only if not tied to a specific forum/process)

SCHEMA (must match exactly):

{
  "version": "Y_inferred_v2",
  "x_tests": {
    "X1": {
      "name": "...",
      "scope": "INTERNAL_DISCIPLINARY_APPEAL|GRIEVANCE|ET|EAT|GENERAL",
      "definition": "...",
      "pattern": "IF <conditions> THEN support X1",
      "required_elements": ["cond1", "cond2", "cond3"],
      "positive_indicators": ["short phrase", "..."],
      "excludes": ["confuser signal", "..."]
    }
  },
  "classes": {
    "DIRECT_X": {"definition": "...", "min_support": 1, "evidence_required": true},
    "CONTRASTIVE": {"definition": "...", "evidence_required": true},
    "REMEDY": {"definition": "...", "evidence_required": true},
    "IRRELEVANT": {"definition": "..."}
  },
  "hard_rules": [
    "rule 1",
    "rule 2",
    "rule 3",
    "rule 4"
  ]
}

CONSTRAINTS:
- x_tests MUST contain between 4 and 7 tests.
- Each test MUST be ATOMIC (one concept each; do NOT mix concepts).
- required_elements MUST contain 3..6 concrete conditions. They should be short, falsifiable, and scorable by code.
- pattern MUST be a single sentence starting with 'IF' and containing 'THEN'.
- positive_indicators MUST be short phrases likely to appear in short summaries.
- excludes MUST list phrases/signals that commonly confuse the test but should NOT count.
- hard_rules MUST be enforceable by code (e.g., evidence requirement, downgrade logic).

Return JSON ONLY.

X:
""".strip()


# =========================
# JSON extraction (robust)
# =========================
def _balanced_json_candidate(s: str) -> str:
    """
    Return the first balanced {...} object in a string.
    Safer than slicing from first '{' to last '}'.
    """
    start = s.find("{")
    if start == -1:
        raise ValueError("No '{' found in model output")

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    raise ValueError("Could not find a balanced JSON object in model output")


def _repair_common_json_issues(candidate: str) -> str:
    c = (candidate or "").strip()

    # Remove markdown fences if present
    c = re.sub(r"^```(?:json)?\s*", "", c, flags=re.IGNORECASE)
    c = re.sub(r"\s*```$", "", c)

    # Normalize smart quotes
    c = c.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # Remove control chars except \n \t \r
    c = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", c)

    # Remove trailing commas before } or ]
    c = re.sub(r",\s*(\}|\])", r"\1", c)

    return c


def extract_json_object(s: str) -> Dict[str, Any]:
    if not s or not s.strip():
        raise ValueError("Model returned empty output.")

    raw = s.strip()

    # Fast path
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    candidate = _balanced_json_candidate(raw)
    candidate = _repair_common_json_issues(candidate)

    try:
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            raise ValueError("Extracted JSON is not an object/dict.")
        return obj
    except json.JSONDecodeError as e:
        # Context window
        lines = candidate.splitlines()
        line_idx = max(0, min(len(lines) - 1, e.lineno - 1))
        preview = "\n".join(lines[max(0, line_idx - 2): min(len(lines), line_idx + 3)])
        pointer = " " * max(0, e.colno - 1) + "^"
        raise ValueError(
            "Near-JSON but failed to parse.\n"
            f"JSONDecodeError: {e.msg} at line {e.lineno} col {e.colno}\n"
            "---- Context (±2 lines) ----\n"
            f"{preview}\n{pointer}\n"
            "---- End Context ----"
        ) from e


# =========================
# Ollama call (JSON mode + retry + debug)
# =========================
def call_ollama(prompt: str, model: str, url: str, timeout_s: int, num_predict: int) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # If supported by your Ollama/model, JSON mode dramatically reduces format drift.
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_predict": num_predict,
            "stop": ["```", "\n\n\n", "</json>", "<END>"],
        },
    }

    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    txt = (data.get("response", "") or "").strip()

    try:
        return extract_json_object(txt)
    except Exception as e:
        _dump_raw("ollama_raw_1", txt)

        # Retry from scratch (do NOT paste broken output)
        repair_prompt = (
            "Return VALID JSON ONLY (no commentary). "
            "Output must start with { and end with }. "
            "Ensure all braces/quotes are balanced. "
            "Do NOT include markdown fences. "
            "End after the final closing brace }.\n\n"
            + prompt
        )
        payload["prompt"] = repair_prompt

        r2 = requests.post(url, json=payload, timeout=timeout_s)
        r2.raise_for_status()
        data2 = r2.json()
        txt2 = (data2.get("response", "") or "").strip()
        _dump_raw("ollama_raw_2", txt2)

        try:
            return extract_json_object(txt2)
        except Exception as e2:
            # Persist parse error message too
            _dump_raw("ollama_parse_error", f"{repr(e)}\n\n{repr(e2)}")
            raise


# =========================
# Quality gates
# =========================
_ALLOWED_SCOPES = {
    "INTERNAL_DISCIPLINARY_APPEAL",
    "GRIEVANCE",
    "ET",
    "EAT",
    "GENERAL",
}

# Words that tend to be vague unless concretized via required_elements.
_VAGUE_TOKENS = {
    "integrity", "fairness", "undermined", "legitimacy", "open-minded",
    "procedural", "procedural integrity", "breach", "failure to maintain",
    "unreasonable", "incompatible", "trust", "confidence", "genuine belief",
}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _contains_vague_language(s: str) -> bool:
    ns = _norm(s)
    return any(v in ns for v in _VAGUE_TOKENS)

def _jaccard(a: List[str], b: List[str]) -> float:
    sa = {_norm(x) for x in a if _norm(x)}
    sb = {_norm(x) for x in b if _norm(x)}
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def _specificity_score(t: Dict[str, Any]) -> float:
    name = str(t.get("name", ""))
    definition = str(t.get("definition", ""))
    scope = str(t.get("scope", "")).strip()
    pattern = str(t.get("pattern", ""))
    req = t.get("required_elements") or []
    pos = t.get("positive_indicators") or []

    score = 0.0

    # Scope reward
    if scope in _ALLOWED_SCOPES:
        score += 3.0
        if scope != "GENERAL":
            score += 1.0

    # Required elements reward
    if isinstance(req, list):
        score += min(3.0, 0.8 * len(req))
        req_text = " ".join([_norm(x) for x in req])
        if "appeal" in req_text or "grievance" in req_text:
            score += 0.75
        if "pending" in req_text or "live" in req_text or "before" in req_text:
            score += 0.75
        if "management" in req_text or "team leader" in req_text:
            score += 0.5

    # Pattern reward
    np = _norm(pattern)
    if np.startswith("if ") and " then " in np:
        score += 1.5
    if "appeal" in np and ("pending" in np or "live" in np or "before" in np):
        score += 0.5

    # Penalize vibe language if not concretized
    vibe = _contains_vague_language(name) or _contains_vague_language(definition)
    if vibe:
        if not isinstance(req, list) or len(req) < 3:
            score -= 4.0
        else:
            score -= 1.5

    # Small reward for concise indicators
    if isinstance(pos, list) and pos:
        score += 0.2 * min(5, len(pos))

    return score


# =========================
# Schema validation (strict + quality)
# =========================
def validate_y_schema_v2(y: Dict[str, Any]) -> None:
    for k in ["version", "x_tests", "classes", "hard_rules"]:
        if k not in y:
            raise ValueError(f"Y missing key: {k}")

    if y.get("version") != "Y_inferred_v2":
        raise ValueError(f"Unexpected version: {y.get('version')} (expected 'Y_inferred_v2')")

    x_tests = y.get("x_tests")
    if not isinstance(x_tests, dict):
        raise ValueError("x_tests must be an object/dict")
    if not (4 <= len(x_tests) <= 7):
        raise ValueError(f"x_tests must contain 4..7 tests, got {len(x_tests)}")

    for key, t in x_tests.items():
        if key not in ("X1", "X2", "X3", "X4", "X5", "X6", "X7"):
            raise ValueError(f"Unexpected test key: {key}")
        if not isinstance(t, dict):
            raise ValueError(f"{key} must be a dict")

        for req in ["name", "scope", "definition", "pattern", "required_elements", "positive_indicators", "excludes"]:
            if req not in t:
                raise ValueError(f"{key} missing '{req}'")

        scope = str(t["scope"]).strip()
        if scope not in _ALLOWED_SCOPES:
            raise ValueError(f"{key}.scope must be one of {_ALLOWED_SCOPES}, got: {scope}")

        pattern = str(t["pattern"]).strip()
        np = _norm(pattern)
        if not (np.startswith("if ") and " then " in np):
            raise ValueError(f"{key}.pattern must start with 'IF' and contain 'THEN'.")

        req_el = t["required_elements"]
        if not isinstance(req_el, list) or not (3 <= len(req_el) <= 6):
            raise ValueError(f"{key}.required_elements must be a list of length 3..6")

        for e in req_el:
            if len(_norm(str(e))) < 8:
                raise ValueError(f"{key}.required_elements contains too-short item: {e}")

        pos = t["positive_indicators"]
        if not isinstance(pos, list) or not pos:
            raise ValueError(f"{key}.positive_indicators must be a non-empty list")
        for p in pos:
            if len(_norm(str(p))) < 3:
                raise ValueError(f"{key}.positive_indicators contains empty/too-short item")

        exc = t["excludes"]
        if not isinstance(exc, list):
            raise ValueError(f"{key}.excludes must be a list")

        if _specificity_score(t) < 2.0:
            raise ValueError(f"{key} failed specificity gate (too vague / too unscoped). name={t.get('name')}")

    classes = y.get("classes")
    if not isinstance(classes, dict):
        raise ValueError("classes must be an object/dict")
    for cname in ["DIRECT_X", "CONTRASTIVE", "REMEDY", "IRRELEVANT"]:
        if cname not in classes:
            raise ValueError(f"classes missing '{cname}'")

    hard_rules = y.get("hard_rules")
    if not isinstance(hard_rules, list) or len(hard_rules) < 4:
        raise ValueError("hard_rules must be a list with at least 4 rules")


# =========================
# Sliding window chunking
# =========================
def sliding_windows(text: str, window_chars: int, stride_chars: int) -> List[str]:
    t = text.strip()
    if not t:
        return []
    if window_chars <= 0:
        return [t]
    if stride_chars <= 0:
        stride_chars = window_chars

    chunks = []
    n = len(t)
    i = 0
    while i < n:
        chunk = t[i:i + window_chars]
        chunks.append(chunk)
        if i + window_chars >= n:
            break
        i += stride_chars
    return chunks


# =========================
# Merge multiple Y drafts (v2: specificity + orthogonality)
# =========================
def merge_y_candidates_v2(
    candidates: List[Dict[str, Any]],
    target_n: int = 6,
    max_overlap: float = 0.55,
    topk_indicators: int = 6,
) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("No Y candidates to merge.")

    def signature(t: Dict[str, Any]) -> str:
        scope = str(t.get("scope", "")).strip()
        req = t.get("required_elements") or []
        req_norm = sorted({_norm(str(x)) for x in req if _norm(str(x))})
        req_key = "|".join(req_norm[:6])
        return f"{scope}::{req_key}"

    pool: List[Dict[str, Any]] = []
    for y in candidates:
        for _, t in (y.get("x_tests") or {}).items():
            if isinstance(t, dict):
                pool.append(t)

    agg: Dict[str, Dict[str, Any]] = {}
    seen_counts: Dict[str, int] = {}

    def extend_unique(dst: List[str], src: List[str], cap: int) -> None:
        for s in src:
            ss = str(s).strip()
            if not ss:
                continue
            if ss not in dst:
                dst.append(ss)
            if len(dst) >= cap:
                break

    for t in pool:
        sig = signature(t)
        if not sig:
            continue
        seen_counts[sig] = seen_counts.get(sig, 0) + 1

        if sig not in agg:
            agg[sig] = {
                "name": t.get("name", ""),
                "scope": t.get("scope", ""),
                "definition": t.get("definition", ""),
                "pattern": t.get("pattern", ""),
                "required_elements": list(t.get("required_elements") or []),
                "positive_indicators": [],
                "excludes": [],
            }

        extend_unique(agg[sig]["positive_indicators"], list(t.get("positive_indicators") or []), topk_indicators)
        extend_unique(agg[sig]["excludes"], list(t.get("excludes") or []), topk_indicators)

        if _specificity_score(t) > _specificity_score(agg[sig]):
            agg[sig]["name"] = t.get("name", agg[sig]["name"])
            agg[sig]["definition"] = t.get("definition", agg[sig]["definition"])
            agg[sig]["pattern"] = t.get("pattern", agg[sig]["pattern"])
            agg[sig]["required_elements"] = list(t.get("required_elements") or agg[sig]["required_elements"])
            agg[sig]["scope"] = t.get("scope", agg[sig]["scope"])

    ranked = sorted(
        agg.items(),
        key=lambda kv: (seen_counts.get(kv[0], 0), _specificity_score(kv[1])),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    for _, t in ranked:
        if len(selected) >= 7:
            break
        overlaps = []
        for s in selected:
            ov = max(
                _jaccard(t.get("required_elements") or [], s.get("required_elements") or []),
                _jaccard(t.get("positive_indicators") or [], s.get("positive_indicators") or []),
            )
            overlaps.append(ov)
        if overlaps and max(overlaps) >= max_overlap:
            continue
        selected.append(t)

    if len(selected) < 4:
        selected = [t for _, t in ranked[:4]]

    keep_n = min(7, max(4, min(target_n, len(selected))))
    selected = selected[:keep_n]

    x_tests: Dict[str, Any] = {}
    for i, t in enumerate(selected, start=1):
        x_tests[f"X{i}"] = t

    merged = {
        "version": "Y_inferred_v2",
        "x_tests": x_tests,
        "classes": candidates[0].get("classes", {}),
        "hard_rules": candidates[0].get("hard_rules", []),
    }
    validate_y_schema_v2(merged)
    return merged


# =========================
# Main: generate Y from WS (v2)
# =========================
def generate_y_from_ws_v2(
    ws_text: str,
    model: str,
    url: str,
    timeout_s: int,
    window_chars: int,
    stride_chars: int,
    num_predict: int,
    merge_topk: int,
    target_n: int,
    max_overlap: float,
) -> Dict[str, Any]:
    chunks = sliding_windows(ws_text, window_chars=window_chars, stride_chars=stride_chars)
    if not chunks:
        raise ValueError("Empty WS text.")

    candidates: List[Dict[str, Any]] = []

    if len(chunks) == 1:
        prompt = Y_PROMPT_INFER_V2 + "\n" + chunks[0]
        y = call_ollama(prompt, model=model, url=url, timeout_s=timeout_s, num_predict=num_predict)
        validate_y_schema_v2(y)
        return y

    for idx, chunk in enumerate(chunks, start=1):
        prompt = (
            Y_PROMPT_INFER_V2
            + "\n"
            + chunk
            + f"\n\n(Chunk {idx}/{len(chunks)}. Infer rubric based ONLY on this chunk.)"
        )
        y = call_ollama(prompt, model=model, url=url, timeout_s=timeout_s, num_predict=num_predict)
        validate_y_schema_v2(y)
        candidates.append(y)

    merged = merge_y_candidates_v2(
        candidates=candidates,
        target_n=target_n,
        max_overlap=max_overlap,
        topk_indicators=merge_topk,
    )
    return merged


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate inferred Y rubric from WS text (Ollama) [v2 hardened].")
    p.add_argument("--ws-file", type=str, default=None, help="Path to a WS text file. If omitted, read from stdin.")
    p.add_argument("--out", type=str, default=None, help="Output JSON path. If omitted, print to stdout only.")
    p.add_argument("--model", type=str, default="mistral-small3.2:latest", help="Ollama model name.")
    p.add_argument("--ollama-url", type=str, default="http://localhost:11434/api/generate", help="Ollama endpoint.")
    p.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds.")
    p.add_argument("--num-predict", type=int, default=2600, help="Max tokens to generate per call (Ollama num_predict).")

    p.add_argument("--window-chars", type=int, default=14000, help="Window size in chars (default: 14000).")
    p.add_argument("--stride-chars", type=int, default=9000, help="Stride in chars (default: 9000).")
    p.add_argument("--no-window", action="store_true", help="Disable windowing (single call).")
    p.add_argument("--merge-topk", type=int, default=6, help="Max indicators/excludes per test kept during merge.")

    p.add_argument("--target-n", type=int, default=6, help="Preferred number of tests to keep (4..7).")
    p.add_argument("--max-overlap", type=float, default=0.55, help="Max allowed overlap between tests (0..1).")

    return p.parse_args()


def read_ws_text(ws_file: Optional[str]) -> str:
    if ws_file:
        return Path(ws_file).read_text(encoding="utf-8", errors="replace")
    import sys
    return sys.stdin.read()


def main() -> None:
    args = parse_args()
    ws_text = read_ws_text(args.ws_file)

    if args.no_window:
        window_chars = 0
        stride_chars = 0
    else:
        window_chars = args.window_chars
        stride_chars = args.stride_chars

    y = generate_y_from_ws_v2(
        ws_text=ws_text,
        model=args.model,
        url=args.ollama_url,
        timeout_s=args.timeout,
        window_chars=window_chars,
        stride_chars=stride_chars,
        num_predict=args.num_predict,
        merge_topk=args.merge_topk,
        target_n=args.target_n,
        max_overlap=args.max_overlap,
    )

    out_json = json.dumps(y, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_json, encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()
