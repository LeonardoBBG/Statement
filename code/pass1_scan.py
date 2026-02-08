#!/usr/bin/env python3
"""
PASS-1: X-matcher over legal reasoning text (summary / reasoning_for_index).

GOAL (airtight, spec-driven, no hidden logic):
- Read Y JSON (must contain x_tests).
- For each appeal record: pick record_text (reasoning_for_index else summary).
- Ask LLM to return STRICT JSON with:
    - matched_X: list of X ids from Y.x_tests keys
    - anchors: list of {"x": "Xk", "start": int, "end": int} spans into record_text
    - confidence: 0..100
    - note: one sentence
- Validate deterministically:
    - matched_X subset of allowed_x
    - each anchor is in-bounds, start < end
    - anchor.x in allowed_x and also in matched_X
    - (optional hard rule) every X in matched_X must have at least one anchor
- Derive verbatim evidence_snippets from spans (not from model text).
- Resume-safe output with stable item_id and append-only JSONL.

No assumptions:
- No hard-coded “core X”.
- No remedy keyword heuristics.
- No model-provided snippet substring checks (we compute verbatim ourselves from spans).
"""

import argparse
import json
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, List, Optional, Set

import requests

# tqdm optional
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


# =========================
# Helpers: text handling
# =========================
_SEG_RE = re.compile(r"(?:\r?\n)?\s*###\s*Segment\s*\d+\s*(?:\r?\n)?", re.IGNORECASE)

def clip(s: Any, n: int = 2500) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    return s[:n]

def clean_segmented_summary(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""
    s = _SEG_RE.sub("\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


# =========================
# JSONL iteration (streaming, multi-appeal safe)
# =========================
def iter_jsonl_records(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def iter_appeals_from_record(rec: Dict[str, Any]) -> Iterator[Tuple[int, Dict[str, Any]]]:
    appeals = rec.get("appeals")
    if isinstance(appeals, list) and appeals:
        for i, a in enumerate(appeals):
            yield i, a if isinstance(a, dict) else {"raw": a}
    else:
        yield 0, rec


# =========================
# Stable ID for resume cache
# =========================
def compute_item_id(src_file: Path, line_obj: Dict[str, Any], appeal_idx: int, appeal_obj: Dict[str, Any]) -> str:
    preferred = (
        appeal_obj.get("uk_eat_no")
        or appeal_obj.get("neutral_citation")
        or appeal_obj.get("case_id")
        or appeal_obj.get("filename")
        or line_obj.get("filename")
        or line_obj.get("source_file")
        or "NA"
    )
    seed = (appeal_obj.get("reasoning_for_index") or appeal_obj.get("summary") or "")[:3000]
    base = f"{src_file.name}::idx={appeal_idx}::pref={preferred}"
    h = hashlib.sha1((base + "||" + seed).encode("utf-8")).hexdigest()[:16]
    return f"{src_file.name}::{appeal_idx}::{preferred}::{h}"


# =========================
# PASS-1: pick record_text field (no assumptions)
# =========================
def pick_record_text(appeal_obj: Dict[str, Any]) -> Tuple[str, str]:
    if appeal_obj.get("reasoning_for_index"):
        return str(appeal_obj["reasoning_for_index"]), "reasoning_for_index"
    if appeal_obj.get("summary"):
        return str(appeal_obj["summary"]), "summary"
    return "", "missing"


# =========================
# Robust JSON extraction from model output
# =========================
def extract_json_object(s: str) -> Dict[str, Any]:
    if not s or not str(s).strip():
        raise ValueError("Model returned empty output.")

    # 1) Try direct json
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) Strip code fences if present
    t = s.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 3) Extract first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object in model output.")

    candidate = s[start:end + 1].strip()
    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)

    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Extracted JSON is not an object/dict.")
    return obj


# =========================
# OLLAMA CALL (strict JSON output)
# =========================
def call_ollama_json(
    prompt: str,
    model: str,
    ollama_url: str,
    timeout_s: int,
    num_predict: int = 450,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict},
    }
    r = requests.post(ollama_url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    txt = (data.get("response", "") or "").strip()
    return extract_json_object(txt)


# =========================
# Y SPEC VALIDATION (no assumptions)
# =========================
def validate_y_spec(y: Dict[str, Any]) -> Tuple[Set[str], List[str]]:
    """
    Returns (allowed_x, issues). If issues non-empty, caller should stop.
    """
    issues: List[str] = []
    if not isinstance(y, dict):
        return set(), ["Y is not a JSON object."]

    x_tests = y.get("x_tests")
    if not isinstance(x_tests, dict) or not x_tests:
        issues.append("Y must contain non-empty object: y['x_tests'].")

    allowed_x: Set[str] = set()
    if isinstance(x_tests, dict):
        for k in x_tests.keys():
            if isinstance(k, str) and k.strip():
                allowed_x.add(k.strip())

    if not allowed_x:
        issues.append("No valid X keys found in y['x_tests'].")

    # Optional: sanity-check X definitions are objects
    if isinstance(x_tests, dict):
        bad_defs = [k for k, v in x_tests.items() if not isinstance(v, dict)]
        if bad_defs:
            issues.append(f"Some x_tests entries are not objects: {bad_defs[:10]}")

    return allowed_x, issues


# =========================
# PROMPT (Pass-1) — X matcher + spans
# =========================
def build_pass1_prompt(record_text: str, y: Dict[str, Any], allowed_x: Set[str], max_text_chars: int) -> str:
    """
    STRICT JSON output. Spans are 0-based indices into record_text.

    We include only the minimum needed from Y:
    - x_tests (definitions) so the model can reason about each X.
    """
    # Keep prompt stable and bounded
    x_tests_compact = y.get("x_tests", {})
    y_compact = json.dumps({"version": y.get("version"), "x_tests": x_tests_compact}, ensure_ascii=False)

    # Limit record_text length deterministically (no assumptions about model context)
    rt = record_text if record_text is not None else ""
    rt = rt[:max_text_chars]

    # Provide explicit instruction that anchors must be inside rt
    allowed_sorted = sorted(list(allowed_x))

    return (
        "You are a legal X-matcher. Return STRICT JSON ONLY. No markdown. No extra keys.\n\n"
        "TASK:\n"
        "Given record_text, identify which X-tests are supported by explicit evidence in record_text.\n"
        "Return matched_X and anchors as character spans (0-based indices) into record_text.\n\n"
        "CRITICAL RULES:\n"
        "1) matched_X MUST be a subset of the allowed X list.\n"
        "2) For every X in matched_X, provide at least ONE anchor span with fields {x,start,end}.\n"
        "3) start/end are character indices into record_text: 0 <= start < end <= len(record_text).\n"
        "4) Do NOT paraphrase. Do NOT invent. Only match if record_text contains evidence.\n"
        "5) If no evidence for any X, return matched_X: [] and anchors: [].\n\n"
        "ALLOWED X IDS:\n"
        f"{json.dumps(allowed_sorted, ensure_ascii=False)}\n\n"
        "OUTPUT SCHEMA (STRICT):\n"
        "{\n"
        '  "matched_X": ["X1","X2"],\n'
        '  "anchors": [\n'
        '    {"x":"X1","start":123,"end":245}\n'
        "  ],\n"
        '  "confidence": 0,\n'
        '  "note": "ONE sentence"\n'
        "}\n\n"
        "X DEFINITIONS (Y):\n"
        f"{y_compact}\n\n"
        "record_text:\n"
        f"{rt}\n"
    )


# =========================
# HARD VALIDATION (spec-driven, deterministic)
# =========================
def normalize_matched_x(matched: Any, allowed_x: Set[str]) -> List[str]:
    if not isinstance(matched, list):
        return []
    out: List[str] = []
    for v in matched:
        if not isinstance(v, str):
            continue
        k = v.strip()
        if k in allowed_x:
            out.append(k)
    # de-dupe, preserve order
    seen = set()
    deduped = []
    for k in out:
        if k not in seen:
            seen.add(k)
            deduped.append(k)
    return deduped

def normalize_anchors(anchors: Any) -> List[Dict[str, Any]]:
    if not isinstance(anchors, list):
        return []
    out: List[Dict[str, Any]] = []
    for a in anchors:
        if not isinstance(a, dict):
            continue
        out.append(a)
    return out

def validate_and_fix_pass1(
    raw_out: Dict[str, Any],
    record_text: str,
    allowed_x: Set[str],
    require_anchor_per_x: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns (fixed_out, actions).
    fixed_out is guaranteed to have keys: matched_X, anchors, confidence, note.
    Additionally returns derived evidence_snippets (computed verbatim) and issues if spans invalid.
    """
    actions: List[str] = []
    out: Dict[str, Any] = {}

    rt = record_text or ""
    n = len(rt)

    matched_X = normalize_matched_x(raw_out.get("matched_X"), allowed_x)
    anchors_raw = normalize_anchors(raw_out.get("anchors"))

    # Normalize confidence/note
    conf = clamp(safe_int(raw_out.get("confidence"), 0), 0, 100)
    note = str(raw_out.get("note") or "").strip()
    note = note[:300] if note else ""

    # Normalize anchors: keep only valid + in-bounds
    anchors_fixed: List[Dict[str, Any]] = []
    for a in anchors_raw:
        x = a.get("x")
        if not isinstance(x, str):
            continue
        x = x.strip()
        if x not in allowed_x:
            actions.append(f"DropAnchor: x not allowed ({x})")
            continue

        start = safe_int(a.get("start"), -1)
        end = safe_int(a.get("end"), -1)

        if start < 0 or end < 0 or start >= end or end > n:
            actions.append(f"DropAnchor: invalid span for {x} (start={start}, end={end}, len={n})")
            continue

        anchors_fixed.append({"x": x, "start": start, "end": end})

    # Rule: anchors must refer only to matched_X
    if anchors_fixed:
        kept: List[Dict[str, Any]] = []
        for a in anchors_fixed:
            if a["x"] in matched_X:
                kept.append(a)
            else:
                actions.append(f"DropAnchor: x not in matched_X ({a['x']})")
        anchors_fixed = kept

    # Rule: if require_anchor_per_x, drop any X lacking anchor
    if require_anchor_per_x and matched_X:
        anchored = {a["x"] for a in anchors_fixed}
        if anchored != set(matched_X):
            missing = [x for x in matched_X if x not in anchored]
            if missing:
                actions.append(f"DropMatchedX: missing anchors for {missing}")
                matched_X = [x for x in matched_X if x in anchored]

    # Derived verbatim snippets (computed, airtight)
    evidence_snippets: List[Dict[str, Any]] = []
    for a in anchors_fixed:
        snippet = rt[a["start"]:a["end"]]
        evidence_snippets.append({"x": a["x"], "snippet": snippet})

    # If model returned junk, we still return a safe object
    out["matched_X"] = matched_X
    out["anchors"] = anchors_fixed
    out["evidence_snippets"] = evidence_snippets
    out["confidence"] = conf if matched_X else 0  # deterministic: no matches => 0
    out["note"] = note or ("No matches." if not matched_X else "")

    return out, actions


# =========================
# RESUME CACHE
# =========================
def load_processed_ids(pass1_out: Path) -> set:
    done = set()
    if not pass1_out.exists():
        return done
    with pass1_out.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                item_id = obj.get("item_id")
                if item_id:
                    done.add(item_id)
            except Exception:
                continue
    return done


# =========================
# RUN PASS-1
# =========================
def run_pass1(
    input_files: List[Path],
    y: Dict[str, Any],
    pass1_out: Path,
    model: str,
    ollama_url: str,
    timeout_s: int,
    debug_max: Optional[int],
    flush_every: int = 100,
    max_text_chars: int = 20000,
    require_anchor_per_x: bool = True,
    llm_retries: int = 1,
) -> int:
    allowed_x, issues = validate_y_spec(y)
    if issues:
        raise ValueError("Invalid Y spec:\n- " + "\n- ".join(issues))

    processed = load_processed_ids(pass1_out)
    pass1_out.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    buf: List[str] = []

    with pass1_out.open("a", encoding="utf-8") as out_f:
        for src in input_files:
            if not src.exists():
                print(f"WARNING missing: {src}")
                continue

            it = iter_jsonl_records(src)
            if tqdm is not None:
                it = tqdm(it, desc=f"scan {src.name}", unit="lines")

            for line_obj in it:
                for appeal_idx, appeal_obj in iter_appeals_from_record(line_obj):
                    item_id = compute_item_id(src, line_obj, appeal_idx, appeal_obj)

                    if item_id in processed:
                        n_skipped += 1
                        if tqdm is not None and hasattr(it, "set_postfix"):
                            try:
                                it.set_postfix(written=n_written, skipped=n_skipped)
                            except Exception:
                                pass
                        continue

                    record_text, source_field = pick_record_text(appeal_obj)

                    base_result = {
                        "item_id": item_id,
                        "source_file": src.name,
                        "appeal_index": appeal_idx,
                        "filename": appeal_obj.get("filename") or line_obj.get("filename"),
                        "neutral_citation": appeal_obj.get("neutral_citation") or line_obj.get("neutral_citation"),
                        "uk_eat_no": appeal_obj.get("uk_eat_no") or line_obj.get("uk_eat_no"),
                        "doc_type": appeal_obj.get("doc_type"),
                        "appeal_type": appeal_obj.get("appeal_type"),
                        "who_appealed": appeal_obj.get("who_appealed"),
                        "outcome": appeal_obj.get("outcome"),
                        "successful": appeal_obj.get("successful"),
                        "favourable_to": appeal_obj.get("favourable_to"),
                        "source_field": source_field,
                        # persist sources for inspection
                        "summary_clean": clean_segmented_summary(appeal_obj.get("summary")),
                        "reasoning_for_index": clip(appeal_obj.get("reasoning_for_index")),
                    }

                    if not (record_text or "").strip():
                        result = {
                            **base_result,
                            "matched_X": [],
                            "anchors": [],
                            "evidence_snippets": [],
                            "confidence": 0,
                            "note": "Missing record_text.",
                            "hard_rule_actions": ["Missing record_text -> no matches"],
                        }
                    else:
                        prompt = build_pass1_prompt(
                            record_text=record_text,
                            y=y,
                            allowed_x=allowed_x,
                            max_text_chars=max_text_chars,
                        )

                        last_err = None
                        raw = None
                        for attempt in range(llm_retries + 1):
                            try:
                                raw = call_ollama_json(
                                    prompt,
                                    model=model,
                                    ollama_url=ollama_url,
                                    timeout_s=timeout_s,
                                    num_predict=450,
                                )
                                last_err = None
                                break
                            except Exception as e:
                                last_err = e

                        if raw is None:
                            # No assumptions: mark as failed, do not invent matches
                            result = {
                                **base_result,
                                "matched_X": [],
                                "anchors": [],
                                "evidence_snippets": [],
                                "confidence": 0,
                                "note": "LLM call failed; no result.",
                                "hard_rule_actions": [f"LLMFailure: {type(last_err).__name__}: {str(last_err)[:200]}"],
                            }
                        else:
                            fixed, actions = validate_and_fix_pass1(
                                raw_out=raw,
                                record_text=record_text[:max_text_chars],
                                allowed_x=allowed_x,
                                require_anchor_per_x=require_anchor_per_x,
                            )

                            result = {
                                **base_result,
                                "matched_X": fixed["matched_X"],
                                "anchors": fixed["anchors"],
                                "evidence_snippets": fixed["evidence_snippets"],
                                "confidence": fixed["confidence"],
                                "note": fixed["note"],
                                "hard_rule_actions": actions,
                            }

                    buf.append(json.dumps(result, ensure_ascii=False))
                    processed.add(item_id)
                    n_written += 1

                    if tqdm is not None and hasattr(it, "set_postfix"):
                        try:
                            it.set_postfix(written=n_written, skipped=n_skipped)
                        except Exception:
                            pass

                    if flush_every and (n_written % flush_every == 0):
                        out_f.write("\n".join(buf) + "\n")
                        out_f.flush()
                        buf.clear()

                    if debug_max and n_written >= debug_max:
                        if buf:
                            out_f.write("\n".join(buf) + "\n")
                            out_f.flush()
                            buf.clear()
                        print("DEBUG_MAX reached.")
                        return n_written

    if buf:
        with pass1_out.open("a", encoding="utf-8") as out_f:
            out_f.write("\n".join(buf) + "\n")
            out_f.flush()

    return n_written


# =========================
# SUMMARY STATS
# =========================
def summarize_pass1(pass1_out: Path, y: Dict[str, Any]) -> None:
    if not pass1_out.exists():
        print("No pass1_results.jsonl found.")
        return

    allowed_x, issues = validate_y_spec(y)
    if issues:
        print("WARNING: Y spec invalid for summary:\n- " + "\n- ".join(issues))
        allowed_x = set()

    counts = {
        "total": 0,
        "missing_text": 0,
        "llm_failed": 0,
        "matched_any": 0,
        "matched_none": 0,
    }
    x_freq = {k: 0 for k in sorted(list(allowed_x))} if allowed_x else {}
    fallback_summary = 0
    top_hits: List[Tuple[int, Dict[str, Any]]] = []

    with pass1_out.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            counts["total"] += 1
            obj = json.loads(line)

            if obj.get("source_field") == "summary":
                fallback_summary += 1
            if obj.get("note") == "Missing record_text.":
                counts["missing_text"] += 1

            actions = obj.get("hard_rule_actions") or []
            if isinstance(actions, list) and any(str(a).startswith("LLMFailure:") for a in actions):
                counts["llm_failed"] += 1

            matched = obj.get("matched_X") or []
            if isinstance(matched, list) and matched:
                counts["matched_any"] += 1
                for x in matched:
                    if x in x_freq:
                        x_freq[x] += 1
                top_hits.append((safe_int(obj.get("confidence"), 0), obj))
            else:
                counts["matched_none"] += 1

    top_hits.sort(key=lambda t: t[0], reverse=True)
    top50 = top_hits[:50]

    print("=== PASS-1 summary ===")
    for k in ["total", "matched_any", "matched_none", "missing_text", "llm_failed"]:
        print(f"{k:14s}: {counts[k]}")

    print(f"\n=== used fallback summary ===\nsummary_used: {fallback_summary}")

    if x_freq:
        print("\n=== matched_X frequency ===")
        for k, v in x_freq.items():
            print(f"{k}: {v}")

    print("\n=== top 50 by confidence (any match) ===")
    for c, obj in top50:
        ident = obj.get("uk_eat_no") or obj.get("neutral_citation") or obj.get("item_id")
        mx = obj.get("matched_X") or []
        print(f"- {c:3d} | {ident} | matched={mx}")


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PASS-1 X-matcher over JSONL corpus using Y x_tests.")
    p.add_argument("--y-json", required=True, help="Path to Y JSON (must contain x_tests).")
    p.add_argument("--input", required=True, nargs="+", help="One or more input JSONL files.")
    p.add_argument("--out", required=True, help="Output JSONL path for pass1 results.")
    p.add_argument("--model", default="mistral-small3.2:latest", help="Ollama model name.")
    p.add_argument("--ollama-url", default="http://localhost:11434/api/generate", help="Ollama generate endpoint.")
    p.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds.")
    p.add_argument("--debug-max", type=int, default=None, help="Cap number of NEW items written.")
    p.add_argument("--no-summary", action="store_true", help="Skip printing summary at end.")
    p.add_argument("--flush-every", type=int, default=100, help="Flush output every N new items (default: 100).")
    p.add_argument("--max-text-chars", type=int, default=20000, help="Truncate record_text to this many chars.")
    p.add_argument("--no-require-anchor-per-x", action="store_true",
                   help="If set, allow matched_X entries even if missing anchors (not recommended).")
    p.add_argument("--llm-retries", type=int, default=1, help="Retries on LLM failure (default: 1).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    y = json.loads(Path(args.y_json).read_text(encoding="utf-8"))
    input_files = [Path(p) for p in args.input]
    pass1_out = Path(args.out)

    # Hard fail if Y invalid (no assumptions)
    allowed_x, issues = validate_y_spec(y)
    if issues:
        raise SystemExit("ERROR: Y spec invalid:\n- " + "\n- ".join(issues))

    n = run_pass1(
        input_files=input_files,
        y=y,
        pass1_out=pass1_out,
        model=args.model,
        ollama_url=args.ollama_url,
        timeout_s=args.timeout,
        debug_max=args.debug_max,
        flush_every=args.flush_every,
        max_text_chars=args.max_text_chars,
        require_anchor_per_x=not args.no_require_anchor_per_x,
        llm_retries=args.llm_retries,
    )
    print(f"Done. New items written: {n}")

    if not args.no_summary:
        summarize_pass1(pass1_out, y)


if __name__ == "__main__":
    main()
