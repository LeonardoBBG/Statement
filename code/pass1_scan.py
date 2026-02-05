#!/usr/bin/env python3
import argparse
import json
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple, List

import requests

# tqdm is optional (no hard dependency)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


# =========================
# JSONL iteration (streaming, multi-appeal safe)
# =========================

_SEG_RE = re.compile(r"(?:\r?\n)?\s*###\s*Segment\s*\d+\s*(?:\r?\n)?", re.IGNORECASE)

def clean_segmented_summary(s: Any) -> str:
    """
    Remove '### Segment N' markers and stitch the remaining text.
    Keeps content, removes the segmentation scaffolding.
    """
    if s is None:
        return ""
    s = str(s).strip()
    if not s:
        return ""

    # Replace segment headers with a newline break
    s = _SEG_RE.sub("\n", s)

    # Collapse excessive whitespace/newlines
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)

    return s.strip()

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
# PASS-1: pick record_text field
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
    if not s or not s.strip():
        raise ValueError("Model returned empty output.")

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object in model output.")

    candidate = s[start:end + 1].strip()
    candidate = re.sub(r"^```(json)?\s*", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s*```$", "", candidate)
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Extracted JSON is not an object/dict.")
    return obj


# =========================
# OLLAMA CALL (strict JSON output)
# =========================
def call_ollama_json(prompt: str, model: str, ollama_url: str, timeout_s: int, num_predict: int = 350) -> Dict[str, Any]:
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
# PROMPT (Pass-1)
# =========================
def build_pass1_prompt(record_text: str, y: Dict[str, Any]) -> str:
    y_compact = json.dumps(y, ensure_ascii=False)
    return (
        "You are a legal classifier. Return STRICT JSON ONLY, no extra text.\n\n"
        "TASK:\n"
        "Classify record_text according to Y rules, and extract ONE evidence_snippet that is a VERBATIM substring of record_text.\n\n"
        "OUTPUT SCHEMA (STRICT):\n"
        '{ "classification": "DIRECT_X|CONTRASTIVE|REMEDY|IRRELEVANT",\n'
        '  "supports_X": ["X1","X2","X3","X4","X5","X6","X7"],\n'
        '  "evidence_snippet": "verbatim snippet from record_text",\n'
        '  "confidence": 0-100,\n'
        '  "note": "ONE sentence"\n'
        "}\n\n"
        "Y:\n"
        f"{y_compact}\n\n"
        "record_text:\n"
        f"{record_text}\n"
    )


# =========================
# HARD VALIDATION LAYER
# =========================
REMEDY_SIGNALS = [
    "polkey", "mitigation", "compensation", "loss", "causation", "timeline",
    "johnson v unisys", "johnson v", "triggs", "remedy", "remedies", "uplift",
    "acaser", "injury to feelings", "schedule of loss"
]

def norm(s: str) -> str:
    return " ".join((s or "").split())

def contains_remedy_only(record_text: str) -> bool:
    t = (record_text or "").lower()
    return any(sig in t for sig in REMEDY_SIGNALS)

def validate_and_fix(raw_out: Dict[str, Any], record_text: str) -> Tuple[Dict[str, Any], List[str]]:
    reasons: List[str] = []
    out = dict(raw_out or {})

    classification = (out.get("classification") or "").strip()
    supports = out.get("supports_X") or []
    if not isinstance(supports, list):
        supports = []
    supports = [str(x).strip() for x in supports if str(x).strip()]

    snippet = out.get("evidence_snippet") or ""
    snippet_n = norm(snippet)
    text_n = norm(record_text)

    if not snippet_n or snippet_n not in text_n:
        out["classification"] = "IRRELEVANT"
        out["supports_X"] = []
        reasons.append("RuleB: evidence_snippet empty or not verbatim found -> IRRELEVANT")
        out["confidence"] = 0
        out["note"] = (out.get("note") or "").strip()[:300]
        out["evidence_snippet"] = (snippet or "").strip()
        return out, reasons

    if classification == "DIRECT_X":
        if not any(x in supports for x in ("X1", "X2", "X3", "X4")):
            if contains_remedy_only(record_text):
                out["classification"] = "REMEDY"
                reasons.append("RuleA: DIRECT_X without X1..X4 -> REMEDY (remedy signals detected)")
            else:
                out["classification"] = "IRRELEVANT"
                reasons.append("RuleA: DIRECT_X without X1..X4 -> IRRELEVANT")
            out["supports_X"] = [x for x in supports if x == "X5"]
            out["confidence"] = 0
            out["note"] = (out.get("note") or "").strip()[:300]
            out["evidence_snippet"] = snippet.strip()
            return out, reasons

    if classification in ("CONTRASTIVE", "DIRECT_X", "REMEDY"):
        if contains_remedy_only(record_text) and not any(x in supports for x in ("X1","X2","X3","X4")):
            out["classification"] = "REMEDY"
            reasons.append("RuleC: remedy-topic signals + no X1..X4 -> REMEDY")

    if out.get("classification") not in ("DIRECT_X", "CONTRASTIVE", "REMEDY", "IRRELEVANT"):
        out["classification"] = "IRRELEVANT"
        out["supports_X"] = []
        reasons.append("Invalid classification value -> IRRELEVANT")

    try:
        c = int(out.get("confidence"))
    except Exception:
        c = 0
    out["confidence"] = max(0, min(100, c))

    allowed = {"X1","X2","X3","X4","X5","X6","X7"}
    out["supports_X"] = [x for x in supports if x in allowed]
    out["note"] = (out.get("note") or "").strip()[:300]
    out["evidence_snippet"] = snippet.strip()
    return out, reasons


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
# RUN PASS-1 (with tqdm)
# =========================
def run_pass1(
    input_files: List[Path],
    y: Dict[str, Any],
    pass1_out: Path,
    model: str,
    ollama_url: str,
    timeout_s: int,
    debug_max: int | None,
    flush_every: int = 100,
) -> int:
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

                    if not record_text.strip():
                        result = {
                            "item_id": item_id,
                            "source_file": src.name,
                            "appeal_index": appeal_idx,
                            "source_field": source_field,

                            # NEW: persist source fields for inspection
                            "summary_clean": clean_segmented_summary(appeal_obj.get("summary")),
                            "reasoning_for_index": clip(appeal_obj.get("reasoning_for_index")),

                            "classification": "IRRELEVANT",
                            "supports_X": [],
                            "evidence_snippet": "",
                            "confidence": 0,
                            "note": "Missing record_text.",
                            "hard_rule_actions": ["Missing record_text -> IRRELEVANT"],
                        }

                    else:
                        prompt = build_pass1_prompt(record_text, y)
                        raw = call_ollama_json(prompt, model=model, ollama_url=ollama_url, timeout_s=timeout_s)
                        fixed, actions = validate_and_fix(raw, record_text)

                        result = {
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

                            # NEW: persist source fields for inspection
                            "summary_clean": clean_segmented_summary(appeal_obj.get("summary")),
                            "reasoning_for_index": clip(appeal_obj.get("reasoning_for_index")),

                            "classification": fixed["classification"],
                            "supports_X": fixed["supports_X"],
                            "evidence_snippet": fixed["evidence_snippet"],
                            "confidence": fixed["confidence"],
                            "note": fixed["note"],
                            "hard_rule_actions": actions,
                        }

                    buf.append(json.dumps(result, ensure_ascii=False))
                    processed.add(item_id)
                    n_written += 1

                    # progress display
                    if tqdm is not None and hasattr(it, "set_postfix"):
                        try:
                            it.set_postfix(written=n_written, skipped=n_skipped)
                        except Exception:
                            pass

                    # batch flush
                    if flush_every and (n_written % flush_every == 0):
                        out_f.write("\n".join(buf) + "\n")
                        out_f.flush()
                        buf.clear()

                    # debug cap (cap on NEW written items)
                    if debug_max and n_written >= debug_max:
                        if buf:
                            out_f.write("\n".join(buf) + "\n")
                            out_f.flush()
                            buf.clear()
                        print("DEBUG_MAX reached.")
                        return n_written

    # final flush
    if buf:
        with pass1_out.open("a", encoding="utf-8") as out_f:
            out_f.write("\n".join(buf) + "\n")
            out_f.flush()

    return n_written


# =========================
# SUMMARY STATS
# =========================
def summarize_pass1(pass1_out: Path) -> None:
    if not pass1_out.exists():
        print("No pass1_results.jsonl found.")
        return

    counts = {"DIRECT_X": 0, "CONTRASTIVE": 0, "REMEDY": 0, "IRRELEVANT": 0}
    x_freq = {f"X{i}": 0 for i in range(1, 8)}
    fallback_summary = 0
    direct_top: List[Tuple[int, Dict[str, Any]]] = []

    with pass1_out.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            cls = obj.get("classification", "IRRELEVANT")
            if cls not in counts:
                cls = "IRRELEVANT"
            counts[cls] += 1

            if obj.get("source_field") == "summary":
                fallback_summary += 1

            for x in obj.get("supports_X") or []:
                if x in x_freq:
                    x_freq[x] += 1

            if cls == "DIRECT_X":
                direct_top.append((int(obj.get("confidence", 0)), obj))

    direct_top.sort(key=lambda t: t[0], reverse=True)
    top50 = direct_top[:50]

    print("=== counts by classification ===")
    for k, v in counts.items():
        print(f"{k:12s}: {v}")

    print("\n=== supports_X frequency ===")
    for k, v in x_freq.items():
        print(f"{k}: {v}")

    print(f"\n=== used fallback summary ===\nsummary_used: {fallback_summary}")

    print("\n=== top 50 DIRECT_X by confidence ===")
    for c, obj in top50:
        ident = obj.get("uk_eat_no") or obj.get("neutral_citation") or obj.get("item_id")
        print(f"- {c:3d} | {ident}")


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pass-1 scanner: classify JSONL corpus using a provided Y JSON.")
    p.add_argument("--y-json", required=True, help="Path to Y JSON (e.g., Y_inferred_v1.json).")
    p.add_argument("--input", required=True, nargs="+", help="One or more input JSONL files.")
    p.add_argument("--out", required=True, help="Output JSONL path for pass1 results.")
    p.add_argument("--model", default="mistral-small3.2:latest", help="Ollama model name.")
    p.add_argument("--ollama-url", default="http://localhost:11434/api/generate", help="Ollama generate endpoint.")
    p.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds.")
    p.add_argument("--debug-max", type=int, default=None, help="Cap number of new items written.")
    p.add_argument("--no-summary", action="store_true", help="Skip printing summary at end.")
    p.add_argument("--flush-every", type=int, default=100,
               help="Flush output every N newly written items (default: 100).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    y = json.loads(Path(args.y_json).read_text(encoding="utf-8"))
    input_files = [Path(p) for p in args.input]
    pass1_out = Path(args.out)

    n = run_pass1(
    input_files=input_files,
    y=y,
    pass1_out=pass1_out,
    model=args.model,
    ollama_url=args.ollama_url,
    timeout_s=args.timeout,
    debug_max=args.debug_max,
    flush_every=args.flush_every,
)
    print(f"Done. New items written: {n}")

    if not args.no_summary:
        summarize_pass1(pass1_out)


if __name__ == "__main__":
    main()
