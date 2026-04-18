"""
y_runner_engine.py  (adapter)

ENGINE ONLY.
- No hardcoded needle/tag vocab.
- No hardcoded tag definitions.
- Notebook supplies ALL calibration inputs: allowed_tags, tag_defs, prompt builder, regex/keywords, etc.

This preserves your notebook logic 1:1, just parameterized.

Deps:
  - pandas
  - tqdm

Usage from Jupyter:
  from y_runner_engine import YRunnerConfig, run_y_pipeline
  df_out, y_results, diag = run_y_pipeline(cfg=..., allowed_tags=..., tag_defs=..., needle_prompt_fn=...)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

@dataclass(frozen=True)
class YRunnerConfig:
    csv_path: Path
    text_col: str

    out_dir: Path
    out_enhanced_csv: Path
    out_y_json: Path

    y_spec_py: Path
    model: str

    debug: bool = False
    debug_slice: str = "3"

    needle_timeout: int = 180
    y_spec_timeout: int = 180

    matches_root: Path = Path(r"/media/hello/Vault/Tribunals/_Matches").resolve()
    master_csv_name: str = "_match_frequencies.csv"
    strict_schema_gate: bool = True

    # NEW:
    #   "master_csv" -> validate against legacy ET master frequency CSV
    #   "tag_spec"   -> validate against _tag_spec.json
    schema_gate_source: str = "master_csv"
    tag_spec_path: Optional[Path] = None


# =========================================================
# HELPERS (stable)
# =========================================================

def _extract_json_object(raw: str) -> str:
    raw = (raw or "").strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object detected in output.")
    return raw[start : end + 1]


def parse_debug_selector(selector: str, n_rows: int) -> list[int]:
    s = (selector or "").strip()
    if not s:
        return list(range(n_rows))

    if s.isdigit():
        k = int(s)
        if k < 1 or k > n_rows:
            raise ValueError(f"DEBUG_SLICE='{s}' out of range 1..{n_rows}")
        return [k - 1]

    if ":" in s:
        parts = s.split(":")
        if len(parts) > 3:
            raise ValueError(f"Invalid slice syntax: '{s}'")

        def to_int(x):
            x = x.strip()
            return None if x == "" else int(x)

        start = to_int(parts[0]) if len(parts) >= 1 else None
        stop = to_int(parts[1]) if len(parts) >= 2 else None
        step = to_int(parts[2]) if len(parts) == 3 else None

        sl = slice(start, stop, step)
        return list(range(n_rows))[sl]

    raise ValueError(f"Unrecognized DEBUG_SLICE format: '{selector}'")


def canonicalize_tag(tag: str, allowed_tags: list[str]) -> str:
    """
    Stable normalization logic; vocab is notebook-supplied.
    Returns canonical tag or "" if unknown.
    """
    if not tag:
        return ""

    t = str(tag).strip().lower()
    t = t.replace("-", "_").replace(" ", "_")
    t = re.sub(r"[^a-z0-9_]+", "", t)
    t = re.sub(r"_+", "_", t).strip("_")

    # NOTE: synonym map is "engine stable"; if you want to calibrate it too,
    # you can fork this function in JN and pass your own canonicalizer.
    synonym_map = {
        "no": "none",
        "null": "none",
        "na": "none",
        "n_a": "none",

        "allow": "upheld",
        "allowed": "upheld",
        "uphold": "upheld",
        "upholding": "upheld",
        "appeal_upheld": "upheld",
        "appeal_allowed": "upheld",

        "verbalwarning": "verbal_warning",
        "verbal_warn": "verbal_warning",
        "verbal_warning": "verbal_warning",
        "verbal": "verbal_warning",

        "no_contemporaneous_record": "no_contemporaneous_evidence",
        "no_contemporaneous_records": "no_contemporaneous_evidence",
        "no_contemporaneous_note": "no_contemporaneous_evidence",
        "no_contemporaneous_notes": "no_contemporaneous_evidence",
        "no_notes": "no_contemporaneous_evidence",
        "no_record": "no_contemporaneous_evidence",
        "no_records": "no_contemporaneous_evidence",
        "no_contemporaneous_evidence": "no_contemporaneous_evidence",

        "pre_determination": "predetermination",
        "pre_determined": "predetermination",
        "predetermined": "predetermination",
        "predetermination": "predetermination",

        "appeal_scope": "appeal_scope_regex",
        "scope_limitation": "appeal_scope_regex",
        "appeal_scope_limit": "appeal_scope_regex",
        "appeal_scope_limitation": "appeal_scope_regex",
        "outside_scope": "appeal_scope_regex",
        "out_of_scope": "appeal_scope_regex",
        "not_in_grounds": "appeal_scope_regex",
        "appeal_scope_regex": "appeal_scope_regex",

        "assumed_intention": "assumed_intention_regex",
        "assumed_intent": "assumed_intention_regex",
        "ulterior_motive": "assumed_intention_regex",
        "improper_motive": "assumed_intention_regex",
        "motive": "assumed_intention_regex",
        "pretext": "assumed_intention_regex",
        "smokescreen": "assumed_intention_regex",
        "sham": "assumed_intention_regex",
        "designed_to_avoid": "assumed_intention_regex",
        "intent_to_avoid": "assumed_intention_regex",
        "assumed_intention_regex": "assumed_intention_regex",

        # computed downstream → ignore if the model outputs it
        "any": "",
        "any_needle": "",
        "match_any": "",
    }

    t = synonym_map.get(t, t)
    return t if t in allowed_tags else ""


# =========================================================
# CORE EXECUTORS (stable)
# =========================================================

def run_y_spec(ws_text: str, *, y_spec_py: Path, model: str, timeout: int) -> tuple[dict | None, str, str, int]:
    try:
        proc = subprocess.run(
            [sys.executable, str(y_spec_py), "--model", model],
            input=ws_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or "").strip()
        stderr = (e.stderr or "TIMEOUT").strip()
        return None, stdout, stderr, 124

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        return None, stdout, stderr, proc.returncode

    try:
        parsed = json.loads(_extract_json_object(stdout))
        return parsed, stdout, stderr, 0
    except Exception:
        return None, stdout, stderr, 0


def run_needle_tagger(
    text: str,
    *,
    model: str,
    timeout: int,
    allowed_tags: list[str],
    canonicalize_fn: Callable[[str, list[str]], str],
    needle_prompt_fn: Callable[[str], str],
) -> tuple[list[dict], str, str, int]:
    """
    Notebook provides needle_prompt_fn(text) → prompt.
    Engine runs ollama and cleans JSON.
    """
    ollama_cmd = ["ollama", "run", model]

    try:
        proc = subprocess.run(
            ollama_cmd,
            input=needle_prompt_fn(text),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or "").strip()
        stderr = (e.stderr or "TIMEOUT").strip()
        return [], stdout, stderr, 124

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        return [], stdout, stderr, proc.returncode

    def _none():
        return ([{"tag": "none", "confidence": 1.0, "negated": False, "evidence_quote": ""}], stdout, stderr, 0)

    try:
        out = json.loads(_extract_json_object(stdout))
        selected = out.get("selected", [])
        if not isinstance(selected, list):
            return _none()

        cleaned: list[dict] = []
        for item in selected:
            if not isinstance(item, dict):
                continue

            tag = canonicalize_fn(item.get("tag"), allowed_tags)
            if not tag:
                continue

            conf = item.get("confidence")
            try:
                conf_f = float(conf) if conf is not None else 0.0
            except Exception:
                conf_f = 0.0
            conf_f = max(0.0, min(1.0, conf_f))

            neg = bool(item.get("negated") is True)
            quote = str(item.get("evidence_quote") or "")[:200]

            cleaned.append({"tag": tag, "confidence": conf_f, "negated": neg, "evidence_quote": quote})

        if not cleaned:
            return _none()

        tags = [d["tag"] for d in cleaned if d.get("tag")]
        if "none" in tags:
            return _none()

        # keep best per tag
        best: dict[str, dict] = {}
        for d in cleaned:
            t = d["tag"]
            if t == "none":
                continue
            if (t not in best) or (d["confidence"] > best[t]["confidence"]):
                best[t] = d

        final = list(best.values()) if best else [{"tag": "none", "confidence": 1.0, "negated": False, "evidence_quote": ""}]
        return final, stdout, stderr, 0

    except Exception:
        return _none()


def flatten_selected(selected: list[dict], allowed_tags: list[str]) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    for t in allowed_tags:
        if t == "none":
            continue
        rec[f"has__{t}"] = False
        rec[f"conf__{t}"] = 0.0
        rec[f"quote__{t}"] = ""

    for item in selected:
        tag = item.get("tag")
        if not tag or tag == "none":
            continue
        if item.get("negated") is True:
            continue
        if f"has__{tag}" not in rec:
            continue

        rec[f"has__{tag}"] = True
        rec[f"conf__{tag}"] = float(item.get("confidence") or 0.0)
        rec[f"quote__{tag}"] = (item.get("evidence_quote") or "")[:200]

    any_hit = any(rec.get(f"has__{t}", False) for t in allowed_tags if t not in ("none", "any_needle"))
    rec["has__any_needle"] = bool(any_hit)
    rec["conf__any_needle"] = 1.0 if any_hit else 0.0
    rec["quote__any_needle"] = ""
    return rec


# =========================================================
# SCHEMA GATE HELPERS
# =========================================================

def _expected_has_columns_from_master_csv(master_freq_path: Path) -> list[str]:
    mf = pd.read_csv(master_freq_path)
    if "match_type" not in mf.columns:
        raise KeyError(f"Master freq table missing 'match_type': {master_freq_path}")

    raw = [str(x).strip() for x in mf["match_type"].dropna().tolist() if str(x).strip()]
    return sorted(set(raw))


def _expected_has_columns_from_tag_spec(tag_spec_path: Path) -> list[str]:
    spec = json.loads(tag_spec_path.read_text(encoding="utf-8"))
    tags = spec.get("tags", [])
    if not isinstance(tags, list):
        raise ValueError(f"Invalid tag spec: missing/invalid 'tags' list: {tag_spec_path}")

    expected: list[str] = []
    for t in tags:
        if not isinstance(t, dict):
            continue
        tag = str(t.get("tag") or "").strip()
        if not tag:
            continue
        expected.append(f"has__{tag}")

    computed = spec.get("computed", [])
    if isinstance(computed, list) and "any_needle" in computed:
        expected.append("has__any_needle")

    return sorted(set(expected))


def validate_enhanced_csv_schema(
    df_out: pd.DataFrame,
    *,
    strict: bool = True,
    source: str,
    master_freq_path: Optional[Path] = None,
    tag_spec_path: Optional[Path] = None,
) -> pd.DataFrame:
    if source == "master_csv":
        if master_freq_path is None:
            raise ValueError("master_freq_path is required when source='master_csv'")
        expected_cols = _expected_has_columns_from_master_csv(master_freq_path)

    elif source == "tag_spec":
        if tag_spec_path is None:
            raise ValueError("tag_spec_path is required when source='tag_spec'")
        expected_cols = _expected_has_columns_from_tag_spec(tag_spec_path)

    else:
        raise ValueError(f"Unsupported schema gate source: {source}")

    expected_set = set(expected_cols)
    out_has_cols = sorted([c for c in df_out.columns if isinstance(c, str) and c.startswith("has__")])
    out_set = set(out_has_cols)

    missing_in_out = sorted(expected_set - out_set)
    extra_in_out = sorted(out_set - expected_set)

    diag_rows = []
    for c in expected_cols:
        diag_rows.append({
            "kind": "expected_has__",
            "col": c,
            "present_in_output": c in out_set,
            "source": source,
        })

    if strict:
        for c in extra_in_out:
            diag_rows.append({
                "kind": "output_extra_has__",
                "col": c,
                "present_in_output": True,
                "source": source,
            })

    diag = pd.DataFrame(diag_rows)

    if missing_in_out:
        raise AssertionError(
            "Enhanced CSV schema mismatch: missing expected has__ columns.\n"
            f"Missing ({len(missing_in_out)}): {missing_in_out}"
        )

    if strict and extra_in_out:
        raise AssertionError(
            "Enhanced CSV schema mismatch: output has__ columns not present in expected schema.\n"
            f"Extra ({len(extra_in_out)}): {extra_in_out}"
        )

    return diag


def validate_enhanced_csv_schema_against_master(
    df_out: pd.DataFrame,
    master_freq_path: Path,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper for legacy callers.
    """
    return validate_enhanced_csv_schema(
        df_out,
        strict=strict,
        source="master_csv",
        master_freq_path=master_freq_path,
        tag_spec_path=None,
    )


# =========================================================
# PIPELINE RUNNER (stable)
# =========================================================

def run_y_pipeline(
    *,
    cfg: YRunnerConfig,
    allowed_tags: list[str],
    tag_defs: dict[str, str],  # kept for notebook parity / prompt closures
    needle_prompt_fn: Callable[[str], str],
    canonicalize_fn: Callable[[str, list[str]], str] = canonicalize_tag,
) -> tuple[pd.DataFrame, dict[str, Any], Optional[pd.DataFrame]]:
    """
    Runs the full notebook pipeline.

    Notebook supplies:
      - allowed_tags
      - tag_defs
      - needle_prompt_fn
      - (optional) canonicalize_fn

    Returns: (df_out, y_results, diag_or_none)
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    master_freq_path = (
        (cfg.matches_root / cfg.master_csv_name).resolve()
        if cfg.schema_gate_source == "master_csv"
        else None
    )

    if cfg.strict_schema_gate:
        if cfg.schema_gate_source == "master_csv":
            if not master_freq_path.exists():
                raise FileNotFoundError(f"Missing ET master freq CSV: {master_freq_path}")

        elif cfg.schema_gate_source == "tag_spec":
            if cfg.tag_spec_path is None:
                raise ValueError("cfg.tag_spec_path is required when schema_gate_source='tag_spec'")
            if not cfg.tag_spec_path.exists():
                raise FileNotFoundError(f"Missing tag spec JSON: {cfg.tag_spec_path}")

        else:
            raise ValueError(f"Unsupported schema_gate_source: {cfg.schema_gate_source}")

    df = pd.read_csv(cfg.csv_path)
    if cfg.text_col not in df.columns:
        raise KeyError(f"Column '{cfg.text_col}' not found. Found: {list(df.columns)}")

    texts_all = df[cfg.text_col].fillna("").astype(str).tolist()
    n_total = len(texts_all)
    idxs = parse_debug_selector(cfg.debug_slice, n_total) if cfg.debug else list(range(n_total))

    print(f"Total rows in CSV: {n_total}")
    print(f"Processing indices (0-based): {idxs[:30]}{' ...' if len(idxs) > 30 else ''}")
    print(f"Count: {len(idxs)} | DEBUG={cfg.debug} | DEBUG_SLICE='{cfg.debug_slice}'")

    enrich_by_idx: dict[int, dict[str, Any]] = {}
    y_results: dict[str, Any] = {
        "version": "Y_inferred_v2",
        "source": {"csv": str(cfg.csv_path), "text_col": cfg.text_col, "model": cfg.model},
        "rows": {}
    }

    for idx in tqdm(idxs, desc="Rows (Needles + y_spec)"):
        ws_text = texts_all[idx].strip()
        X1 = idx + 1

        if not ws_text:
            empty_selected = [{"tag": "none", "confidence": 1.0, "negated": False, "evidence_quote": ""}]
            rec: dict[str, Any] = {
                "ws_len": 0,
                "needle_selected_raw": json.dumps(empty_selected, ensure_ascii=False),
                "needle_rc": 0,
                "y_ok": False,
                "y_rc": 0,
            }
            rec.update(flatten_selected(empty_selected, allowed_tags))
            enrich_by_idx[idx] = rec
            continue

        print(f"X1={X1} doc={cfg.csv_path.name} | stage=needle")

        t0 = time.time()
        selected, n_stdout, n_stderr, n_rc = run_needle_tagger(
            ws_text,
            model=cfg.model,
            timeout=cfg.needle_timeout,
            allowed_tags=allowed_tags,
            canonicalize_fn=canonicalize_fn,
            needle_prompt_fn=needle_prompt_fn,
        )
        needle_secs = round(time.time() - t0, 2)

        print(f"X1={X1} doc={cfg.csv_path.name} | stage=y_spec | needle_secs={needle_secs}")

        t1 = time.time()
        y_json, y_stdout, y_stderr, y_rc = run_y_spec(
            ws_text,
            y_spec_py=cfg.y_spec_py,
            model=cfg.model,
            timeout=cfg.y_spec_timeout,
        )
        y_secs = round(time.time() - t1, 2)

        y_results["rows"][f"X1_{X1:04d}"] = {
            "row_index_1based": X1,
            "doc": cfg.csv_path.name,
            "y_ok": bool(y_json),
            "y": y_json if y_json else None,
            "y_returncode": y_rc,
            "y_stderr_head": (y_stderr[:400] if cfg.debug and y_stderr else ""),
            "y_stdout_head": (y_stdout[:400] if cfg.debug and y_stdout else ""),
            "timing": {"needle_secs": needle_secs, "y_secs": y_secs},
        }

        rec = {
            "ws_len": len(ws_text),
            "needle_selected_raw": json.dumps(selected, ensure_ascii=False),
            "needle_rc": n_rc,
            "y_ok": bool(y_json),
            "y_rc": y_rc,
        }
        rec.update(flatten_selected(selected, allowed_tags))
        enrich_by_idx[idx] = rec

    enrich_df = pd.DataFrame.from_dict(enrich_by_idx, orient="index")
    enrich_df.index.name = "row_index_0based"

    df_out = df.copy().join(enrich_df, how="left")
    df_out.insert(0, "X1", df_out.index + 1)
    df_out.insert(1, "doc", cfg.csv_path.name)

    df_out.to_csv(cfg.out_enhanced_csv, index=False)
    cfg.out_y_json.write_text(json.dumps(y_results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nWrote outputs:")
    print(f" - Enhanced CSV: {cfg.out_enhanced_csv}")
    print(f" - Y_inferred.json: {cfg.out_y_json}")

    diag = None
    if cfg.strict_schema_gate:
        print(f"\n[gate] Validating enhanced needle columns using source='{cfg.schema_gate_source}'...")

        diag = validate_enhanced_csv_schema(
            df_out,
            strict=True,
            source=cfg.schema_gate_source,
            master_freq_path=master_freq_path if cfg.schema_gate_source == "master_csv" else None,
            tag_spec_path=cfg.tag_spec_path if cfg.schema_gate_source == "tag_spec" else None,
        )

        print("[gate] OK — schema aligned.")

    return df_out, y_results, diag