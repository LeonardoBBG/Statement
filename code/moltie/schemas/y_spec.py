#!/usr/bin/env python3
"""
generate_y_from_ws.py

Generate an inferred rubric Y (Y_inferred_v1) from a Witness Statement (WS) text using Ollama.

- Input: WS text (from a file or stdin)
- Output: Y JSON (to stdout and/or file)
- Token/length safety: optional sliding-window inference over WS chunks, then merge into one Y.

Usage examples:
  python generate_y_from_ws.py --ws-file ./ws.txt --out ./Y_inferred.json
  cat ws.txt | python generate_y_from_ws.py --out ./Y_inferred.json
  python generate_y_from_ws.py --ws-file ./ws.txt --window-chars 14000 --stride-chars 9000 --merge-topk 5
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


# =========================
# PROMPT: FREE INFERENCE Y
# =========================
Y_PROMPT_INFER = """
You are to INFER an evaluation rubric Y from the fact pattern X.

OUTPUT:
Return VALID JSON ONLY (no markdown, no code fences, no commentary).
The output must start with { and end with }.

GOAL:
Create 4 to 7 ATOMIC tests that capture the core factual/legal signals in X, such that a short case description
("reasoning_for_index" or "summary") can be classified for usefulness.

SCHEMA (must match exactly):

{
  "version": "Y_inferred_v1",
  "x_tests": {
    "X1": {"name": "...", "definition": "...", "positive_indicators": ["..."], "excludes": ["..."]},
    "X2": {"name": "...", "definition": "...", "positive_indicators": ["..."], "excludes": ["..."]},
    "X3": {"name": "...", "definition": "...", "positive_indicators": ["..."], "excludes": ["..."]},
    "X4": {"name": "...", "definition": "...", "positive_indicators": ["..."], "excludes": ["..."]}
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
- positive_indicators MUST be short phrases likely to appear in short summaries.
- excludes MUST list phrases/signals that commonly confuse the test but should NOT count.
- DIRECT_X = supports at least one test that reflects the core wrongdoing/fairness/confidentiality themes implied by X.
- CONTRASTIVE = describes correct/proper handling that contrasts with X, but does NOT match DIRECT_X tests.
- REMEDY = primarily about compensation/mitigation/Polkey/causation/timeline.
- IRRELEVANT = none of the above.
- hard_rules MUST be enforceable by code (e.g., snippet evidence requirement, min_support thresholds, downgrade logic).

Return JSON ONLY.

X:
""".strip()


# =========================
# JSON extraction (robust)
# =========================
def extract_json_object(s: str) -> Dict[str, Any]:
    if not s or not s.strip():
        raise ValueError("Model returned empty output.")

    # Fast path: already JSON dict
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find a JSON object in model output.")

    candidate = s[start:end + 1].strip()
    candidate = re.sub(r"^```(json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)

    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Extracted JSON is not an object/dict.")
    return obj


# =========================
# Ollama call
# =========================
def call_ollama(prompt: str, model: str, url: str, timeout_s: int, num_predict: int) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": num_predict,
        },
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    txt = (data.get("response", "") or "").strip()
    return extract_json_object(txt)


# =========================
# Basic schema validation (strict)
# =========================
def validate_y_schema(y: Dict[str, Any]) -> None:
    for k in ["version", "x_tests", "classes", "hard_rules"]:
        if k not in y:
            raise ValueError(f"Y missing key: {k}")

    if y.get("version") != "Y_inferred_v1":
        raise ValueError(f"Unexpected version: {y.get('version')} (expected 'Y_inferred_v1')")

    x_tests = y.get("x_tests")
    if not isinstance(x_tests, dict):
        raise ValueError("x_tests must be an object/dict")
    if not (4 <= len(x_tests) <= 7):
        raise ValueError(f"x_tests must contain 4..7 tests, got {len(x_tests)}")

    # Minimal structure checks per test
    for key, t in x_tests.items():
        if key not in ("X1", "X2", "X3", "X4", "X5", "X6", "X7"):
            raise ValueError(f"Unexpected test key: {key}")
        if not isinstance(t, dict):
            raise ValueError(f"{key} must be a dict")
        for req in ["name", "definition", "positive_indicators", "excludes"]:
            if req not in t:
                raise ValueError(f"{key} missing '{req}'")
        if not isinstance(t["positive_indicators"], list) or not t["positive_indicators"]:
            raise ValueError(f"{key}.positive_indicators must be a non-empty list")
        if not isinstance(t["excludes"], list):
            raise ValueError(f"{key}.excludes must be a list")

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
# Merge multiple Y drafts (simple, deterministic)
# =========================
def merge_y_candidates(candidates: List[Dict[str, Any]], topk_indicators: int = 6) -> Dict[str, Any]:
    """
    Deterministic merge:
    - Collect all tests across candidates
    - Rank tests by "indicator richness" (len(positive_indicators)+len(excludes)) then by presence across candidates
    - Keep up to 7 tests; renumber as X1..Xn
    - Keep classes + hard_rules from the first candidate (they should be similar)
    """
    if not candidates:
        raise ValueError("No Y candidates to merge.")

    # Build signature map for "same-ish" tests by normalized name
    def norm_name(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    pool: List[Tuple[str, Dict[str, Any]]] = []
    for y in candidates:
        for _, t in (y.get("x_tests") or {}).items():
            if isinstance(t, dict):
                pool.append((norm_name(t.get("name", "")), t))

    # Aggregate by name
    agg: Dict[str, Dict[str, Any]] = {}
    counts: Dict[str, int] = {}
    richness: Dict[str, int] = {}

    for name_key, t in pool:
        if not name_key:
            continue
        counts[name_key] = counts.get(name_key, 0) + 1

        pos = [str(x).strip() for x in (t.get("positive_indicators") or []) if str(x).strip()]
        exc = [str(x).strip() for x in (t.get("excludes") or []) if str(x).strip()]

        if name_key not in agg:
            agg[name_key] = {
                "name": t.get("name", ""),
                "definition": t.get("definition", ""),
                "positive_indicators": [],
                "excludes": [],
            }

        # Merge indicators (preserve order, dedupe)
        def extend_unique(dst: List[str], src: List[str], cap: int) -> None:
            for s in src:
                if s not in dst:
                    dst.append(s)
                if len(dst) >= cap:
                    break

        extend_unique(agg[name_key]["positive_indicators"], pos, topk_indicators)
        extend_unique(agg[name_key]["excludes"], exc, topk_indicators)

        richness[name_key] = len(agg[name_key]["positive_indicators"]) + len(agg[name_key]["excludes"])

    # Rank: higher count across windows, then richness
    ranked = sorted(
        agg.items(),
        key=lambda kv: (counts.get(kv[0], 0), richness.get(kv[0], 0)),
        reverse=True,
    )

    # Keep 4..7 tests (prefer 6 if available)
    keep_n = min(7, max(4, min(6, len(ranked))))
    kept = ranked[:keep_n]

    x_tests: Dict[str, Any] = {}
    for i, (_, t) in enumerate(kept, start=1):
        x_tests[f"X{i}"] = t

    merged = {
        "version": "Y_inferred_v1",
        "x_tests": x_tests,
        "classes": candidates[0].get("classes", {}),
        "hard_rules": candidates[0].get("hard_rules", []),
    }
    validate_y_schema(merged)
    return merged


# =========================
# Main: generate Y from WS
# =========================
def generate_y_from_ws(
    ws_text: str,
    model: str,
    url: str,
    timeout_s: int,
    window_chars: int,
    stride_chars: int,
    num_predict: int,
    merge_topk: int,
) -> Dict[str, Any]:
    chunks = sliding_windows(ws_text, window_chars=window_chars, stride_chars=stride_chars)
    if not chunks:
        raise ValueError("Empty WS text.")

    candidates: List[Dict[str, Any]] = []

    # If single chunk, one call.
    if len(chunks) == 1:
        prompt = Y_PROMPT_INFER + "\n" + chunks[0]
        y = call_ollama(prompt, model=model, url=url, timeout_s=timeout_s, num_predict=num_predict)
        validate_y_schema(y)
        return y

    # Multi-chunk: infer Y per chunk, then merge
    for idx, chunk in enumerate(chunks, start=1):
        prompt = (
            Y_PROMPT_INFER
            + "\n"
            + chunk
            + f"\n\n(Chunk {idx}/{len(chunks)}. Infer rubric based ONLY on this chunk.)"
        )
        y = call_ollama(prompt, model=model, url=url, timeout_s=timeout_s, num_predict=num_predict)
        validate_y_schema(y)
        candidates.append(y)

    merged = merge_y_candidates(candidates, topk_indicators=merge_topk)
    return merged


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate inferred Y rubric from WS text (Ollama).")
    p.add_argument("--ws-file", type=str, default=None, help="Path to a WS text file. If omitted, read from stdin.")
    p.add_argument("--out", type=str, default=None, help="Output JSON path. If omitted, print to stdout only.")
    p.add_argument("--model", type=str, default="mistral-small3.2:latest", help="Ollama model name.")
    p.add_argument("--ollama-url", type=str, default="http://localhost:11434/api/generate", help="Ollama endpoint.")
    p.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds.")
    p.add_argument("--num-predict", type=int, default=900, help="Max tokens to generate per call (Ollama num_predict).")

    # Sliding window safety (character-based to avoid tokenizer dependency)
    p.add_argument("--window-chars", type=int, default=14000, help="Window size in chars (default: 14000).")
    p.add_argument("--stride-chars", type=int, default=9000, help="Stride in chars (default: 9000).")
    p.add_argument("--no-window", action="store_true", help="Disable windowing (single call).")
    p.add_argument("--merge-topk", type=int, default=6, help="Max indicators/excludes per test kept during merge.")

    return p.parse_args()


def read_ws_text(ws_file: Optional[str]) -> str:
    if ws_file:
        return Path(ws_file).read_text(encoding="utf-8", errors="replace")
    # stdin
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

    y = generate_y_from_ws(
        ws_text=ws_text,
        model=args.model,
        url=args.ollama_url,
        timeout_s=args.timeout,
        window_chars=window_chars,
        stride_chars=stride_chars,
        num_predict=args.num_predict,
        merge_topk=args.merge_topk,
    )

    out_json = json.dumps(y, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_json, encoding="utf-8")
    print(out_json)


if __name__ == "__main__":
    main()