import sys
import json
import importlib
import re
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st
from needle_bridge import NeedleBridgeConfig, build_needle_run_plan

# ==========================================================
# MOLTIE RUNNER APP (DUAL-MODE / PDF-FIRST / RESUME-FIXED)
# - loads grouped_jobs_df from disk
# - supports offensive / defensive precedent universes
# - lets user select jobs by row_id + precedent_mode
# - shows dynamic output path
# - optional execution with resume mode
# - resume logic at atomic level
# - CSV parsing for nested columns
# - execution grouped by PDF within the selected row_ids only
# - progress bar tracks PDFs, not atoms
# ==========================================================

# -------------------------
# Constants / defaults
# -------------------------
REPO_ROOT = Path("/home/hello/Projects/Statements").resolve()
DEFAULT_OUT_DIR = REPO_ROOT / "output" / "moltie_batch"
DEFAULT_GROUPED_JOBS_PATH = REPO_ROOT / "output" / "grouped_jobs_df_all_modes.parquet"
DEFAULT_CSV_GROUPED_JOBS_PATH = DEFAULT_OUT_DIR / "grouped_jobs_csv_bridge.parquet"
DEFAULT_MANUAL_GROUPED_JOBS_PATH = DEFAULT_OUT_DIR / "grouped_jobs_manual_y.parquet"

CODE_ROOT = (REPO_ROOT / "code").resolve()
ADAPTER_ROOT = (CODE_ROOT / "adapter").resolve()
DEFAULT_Y_PATH = REPO_ROOT / "output" / "Y_inferred.json"
DEFAULT_Y_MANUAL_PATH = REPO_ROOT / "output" / "debug_y" / "Y_manual.json"
DEFAULT_Y_BY_MODE = {
    "offensive": REPO_ROOT / "output" / "Y_inferred_offensive.json",
    "defensive": REPO_ROOT / "output" / "Y_inferred_defensive.json",
}
BASE_MATCHES_ROOT = Path("/media/hello/Vault/Tribunals/_Matches").resolve()
ET_INPUT_ROOT = Path("/media/hello/Vault/Tribunals/ET_Cases").resolve()
NEEDLES_INPUT_ROOT = (REPO_ROOT / "input" / "needles").resolve()
DEFAULT_WS_INPUT_CSV = REPO_ROOT / "input" / "Leonardo_WS_copy.csv"
DEFAULT_WS_TEXT_COL = "text_verbatim"
DEFAULT_Y_SPEC_PY = REPO_ROOT / "code" / "moltie" / "schemas" / "y_spec.py"

DEFAULT_MATCHES_INDEX_BY_MODE = {
    "offensive": BASE_MATCHES_ROOT / "offensive" / "_matches_index.csv",
    "defensive": BASE_MATCHES_ROOT / "defensive" / "_matches_index.csv",
}

DEFAULT_WS_ENHANCED_BY_MODE = {
    "offensive": REPO_ROOT / "output" / "Leonardo_WS_enhanced_offensive.csv",
    "defensive": REPO_ROOT / "output" / "Leonardo_WS_enhanced_defensive.csv",
}

DEFAULT_NEEDLE_JSON_BY_MODE = {
    "offensive": NEEDLES_INPUT_ROOT / "offensive_needles.json",
    "defensive": NEEDLES_INPUT_ROOT / "defensive_needles.json",
}

DEFAULT_MODEL = "mistral-small3.2:latest"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

WORKFLOW_CONFIG = {
    "CSV Bridge": {
        "grouped_path": DEFAULT_CSV_GROUPED_JOBS_PATH,
        "y_path": DEFAULT_Y_PATH,
        "caption": "Legacy CSV bridge: ET matches + enhanced CSV + mode-specific row-derived Y.",
    },
    "Manual JSON": {
        "grouped_path": DEFAULT_MANUAL_GROUPED_JOBS_PATH,
        "y_path": DEFAULT_Y_MANUAL_PATH,
        "caption": "Manual Y workflow: grouped corpus built directly from output/debug_y/Y_manual.json.",
    },
}


# ==========================================================
# Helpers
# ==========================================================
def as_list(x):
    """Safely normalize scalar/list/np-array/arrow-list/pandas objects into a Python list."""
    if x is None:
        return []
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass

    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass

    if isinstance(x, (list, tuple, set)):
        return list(x)

    return [x]


def _parse_maybe_jsonlike(x):
    """
    Parse values that may be JSON strings or Python-literal strings.
    Useful for CSV round-trips of list/dict columns.
    """
    if x is None:
        return None

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, (list, tuple, set, dict)):
        return x

    if not isinstance(x, str):
        return x

    s = x.strip()
    if not s:
        return None

    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        import ast
        return ast.literal_eval(s)
    except Exception:
        return x


def _safe_token(s: str) -> str:
    s = str(s or "").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch in {"_", "-"}:
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "unknown"


def _safe_row_id_list_tag(idxs: List[Any], max_items: int = 20) -> str:
    """
    Build a compact tag for numeric or string row ids. Truncates if too many.
    """
    if not idxs:
        return "Y_NONE"
    idxs = [str(i).strip() for i in idxs if str(i).strip()]
    if not idxs:
        return "Y_NONE"
    if len(idxs) == 1:
        return f"Y_{_safe_token(idxs[0])}"
    head = idxs[:max_items]
    tail_n = len(idxs) - len(head)
    base = "Y_" + "_".join(_safe_token(i) for i in head)
    if tail_n > 0:
        base += f"__plus_{tail_n}"
    return base


def _safe_mode_tag(modes: List[str]) -> str:
    modes = sorted({_safe_token(m) for m in modes if str(m).strip()})
    if not modes:
        return "mode_none"
    if len(modes) == 1:
        return f"mode_{modes[0]}"
    return "mode_multi_" + "_".join(modes)


def _ensure_sys_path():
    assert CODE_ROOT.exists(), f"Missing CODE_ROOT: {CODE_ROOT}"
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    if str(ADAPTER_ROOT) not in sys.path:
        sys.path.insert(0, str(ADAPTER_ROOT))


@st.cache_data(show_spinner=False)
def load_grouped_jobs(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"grouped_jobs_df not found: {p}")

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError("Unsupported file type. Use .parquet or .csv")

    for col in ["matched_needles", "x_tests"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_maybe_jsonlike)

    if "matched_needles" in df.columns:
        df["matched_needles"] = df["matched_needles"].apply(as_list)

    if "x_tests" in df.columns:
        def _normalize_x_tests(x):
            vals = as_list(x)
            out = []
            for item in vals:
                if isinstance(item, dict):
                    out.append(item)
            return out

        df["x_tests"] = df["x_tests"].apply(_normalize_x_tests)

    if "precedent_mode" not in df.columns:
        df["precedent_mode"] = "unknown"

    return df


@st.cache_data(show_spinner=False)
def load_y_rows(path: str) -> dict:
    y_path = Path(path)
    assert y_path.exists(), f"Missing Y: {y_path}"
    y_root = json.loads(y_path.read_text(encoding="utf-8"))
    rows = y_root.get("rows") or {}
    assert isinstance(rows, dict) and rows, "Y has no rows"
    return rows


def build_output_path(
    out_dir: Path,
    selected_row_ids: List[Any],
    selected_precedent_modes: List[str],
    debug: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    row_tag = _safe_row_id_list_tag(selected_row_ids)
    mode_tag = _safe_mode_tag(selected_precedent_modes)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_mode = "DEBUG" if debug else "BATCH"
    return out_dir / f"batch_results_{run_mode}_{mode_tag}_{row_tag}_{ts}.jsonl"


def safe_to_dict(obj):
    if obj is None:
        return None
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return {"repr": repr(obj)}


def _load_needle_pack(mode: str) -> dict[str, Any]:
    pack_path = DEFAULT_NEEDLE_JSON_BY_MODE[mode]
    if not pack_path.exists():
        raise FileNotFoundError(f"Missing needle pack: {pack_path}")
    return json.loads(pack_path.read_text(encoding="utf-8"))


def _compile_regex_buckets(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], str]:
    flag_map = {
        "IGNORECASE": re.IGNORECASE,
        "VERBOSE": re.VERBOSE,
        "MULTILINE": re.MULTILINE,
        "DOTALL": re.DOTALL,
    }
    compiled = {}
    folder_map = {}
    regex_only_folder_name = "_REGEX_ONLY"

    for bucket_name, bucket_cfg in (raw or {}).items():
        if not isinstance(bucket_cfg, dict) or not bucket_cfg.get("pattern"):
            continue
        flags = 0
        for f_name in bucket_cfg.get("flags", []):
            flags |= flag_map.get(str(f_name).upper().strip(), 0)
        compiled[bucket_name] = re.compile(bucket_cfg["pattern"], flags)
        folder_map[bucket_name] = bucket_cfg.get("folder_name", f"_{str(bucket_name).upper()}")
        regex_only_folder_name = bucket_cfg.get("regex_only_folder_name", regex_only_folder_name)

    return compiled, folder_map, regex_only_folder_name


def _canonicalize_tag_local(tag: str) -> str:
    s = str(tag).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _compile_tag_spec_from_needles_json(mode: str) -> tuple[dict[str, Any], Path]:
    source_path = DEFAULT_NEEDLE_JSON_BY_MODE[mode]
    matches_root = (BASE_MATCHES_ROOT / mode).resolve()
    matches_root.mkdir(parents=True, exist_ok=True)

    payload = _load_needle_pack(mode)
    needles_all = payload.get("needles_all", [])
    needles_any = payload.get("needles_any", [])
    regex_buckets = payload.get("regex_buckets", {})

    reserved = {"any_needle", "none"}
    tags = []
    seen = set()

    for raw_tag in needles_any:
        if not isinstance(raw_tag, str):
            continue
        tag = _canonicalize_tag_local(raw_tag)
        if not tag or tag in reserved or tag in seen:
            continue
        seen.add(tag)
        tags.append(
            {
                "tag": tag,
                "type": "needle_any",
                "source": {"phrase": raw_tag},
                "desc": f"keyword/phrase tag: '{raw_tag}'",
                "corpus_column": f"has__{tag}",
            }
        )

    for bucket_name, bucket_meta in regex_buckets.items():
        if not isinstance(bucket_name, str) or not isinstance(bucket_meta, dict) or not bucket_meta.get("pattern"):
            continue
        tag = _canonicalize_tag_local(bucket_name)
        if not tag or tag in reserved or tag in seen:
            continue
        seen.add(tag)
        tags.append(
            {
                "tag": tag,
                "type": "regex",
                "source": {"name": bucket_name},
                "desc": f"regex bucket: {tag.replace('_', ' ')}",
                "corpus_column": f"has__{tag}",
            }
        )

    spec = {
        "version": "v1",
        "compiled_from": str(source_path),
        "compiled_mode": mode,
        "needles_all_gate": needles_all,
        "computed": ["any_needle", "none"],
        "tags": tags,
    }

    out_path = matches_root / "_tag_spec.json"
    out_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return spec, out_path


def _build_allowed_tags_and_defs_from_spec(spec: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    tag_list = []
    seen = set()
    for t in spec.get("tags", []):
        tag = t.get("tag")
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tag_list.append(tag)

    computed = spec.get("computed", [])
    allowed = []
    if "any_needle" in computed:
        allowed.append("any_needle")
    allowed.extend(tag_list)
    if "none" in computed:
        allowed.append("none")

    defs = {}
    for t in spec.get("tags", []):
        tag = t.get("tag")
        if tag:
            defs[tag] = (t.get("desc") or f"tag: {tag}").strip()
    if "any_needle" in computed:
        defs["any_needle"] = "computed: at least one non-none tag applies"
    if "none" in computed:
        defs["none"] = "none of the above apply"
    return allowed, defs


def _make_needle_prompt_builder(allowed_tags: list[str], tag_defs: dict[str, str]) -> Callable[[str], str]:
    def make_needle_prompt(text: str) -> str:
        model_allowed = [t for t in allowed_tags if t != "any_needle"]
        tag_lines = "\n".join([f'- "{t}": {tag_defs.get(t, "")}' for t in model_allowed])
        return f"""
TASK:
Given TEXT, select all applicable TAGS from the allowed list.

CRITICAL TAG RULE:
- Each "tag" value MUST be EXACTLY one of the strings in ALLOWED_TAGS.
- Do NOT invent new tag names.
- Do NOT output "any_needle".
- If uncertain, return ONLY ["none"].

ALLOWED_TAGS:
{tag_lines}

RULES:
- For every selected tag (except "none"), provide confidence, negation, and evidence_quote.
- If you cannot quote evidence from TEXT, do NOT select that tag.
- Return VALID JSON ONLY.

TEXT:
<<<
{text.strip()}
>>>

OUTPUT JSON SCHEMA:
{{
  "selected": [
    {{"tag": "...", "confidence": 0.0, "negated": false, "evidence_quote": "..."}}
  ]
}}
""".strip()

    return make_needle_prompt


def _pack_tests(g: pd.DataFrame) -> list[dict]:
    seen = set()
    out = []
    for _, r in g.iterrows():
        k = r["x_key"]
        if k in seen:
            continue
        seen.add(k)
        out.append(
            {
                "x_key": k,
                "x_name": r["x_name"],
                "x_scope": r["x_scope"],
            }
        )
    return out


def _build_grouped_jobs(run_plan_df: pd.DataFrame) -> pd.DataFrame:
    if run_plan_df.empty:
        return pd.DataFrame(
            columns=[
                "row_id",
                "y_row_id",
                "et_path",
                "precedent_mode",
                "match_mode",
                "matched_needles",
                "x_tests",
            ]
        )

    return (
        run_plan_df
        .groupby(["row_id", "y_row_id", "et_path", "precedent_mode", "match_mode"], dropna=False)
        .apply(
            lambda g: pd.Series(
                {
                    "matched_needles": sorted(set(g["matched_needle"].dropna())),
                    "x_tests": _pack_tests(g),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )


def _extract_x_tests(payload: dict) -> list[dict]:
    if not isinstance(payload, dict):
        return []

    raw = payload.get("x_tests") or {}
    out = []

    if isinstance(raw, dict):
        for x_key, x_obj in raw.items():
            x_obj = x_obj or {}
            out.append(
                {
                    "x_key": x_key,
                    "x_name": x_obj.get("name", x_key),
                    "x_scope": x_obj.get("scope", "GENERAL"),
                }
            )
        return out

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            x_key = item.get("x_key") or item.get("key") or item.get("id")
            if not x_key:
                continue
            out.append(
                {
                    "x_key": x_key,
                    "x_name": item.get("x_name") or item.get("name") or x_key,
                    "x_scope": item.get("x_scope") or item.get("scope") or "GENERAL",
                }
            )

    return out


def _iter_manual_y_records(path: Path):
    root = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(root, dict) and isinstance(root.get("rows"), dict):
        for y_row_id, row_obj in root["rows"].items():
            yield y_row_id, row_obj or {}
        return

    if isinstance(root, dict) and isinstance(root.get("items"), list):
        for i, row_obj in enumerate(root["items"], start=1):
            row_obj = row_obj or {}
            y_row_id = row_obj.get("y_row_id") or row_obj.get("id") or f"MANUAL_{i:04d}"
            yield y_row_id, row_obj
        return

    if isinstance(root, list):
        for i, row_obj in enumerate(root, start=1):
            row_obj = row_obj or {}
            y_row_id = row_obj.get("y_row_id") or row_obj.get("id") or f"MANUAL_{i:04d}"
            yield y_row_id, row_obj
        return

    raise ValueError("Manual Y JSON must be either {'rows': {...}}, {'items': [...]}, or a top-level list.")


def _as_path_list(value) -> list[str]:
    vals = as_list(value)
    out = []
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return out


def _extract_et_paths(row_obj: dict, payload: dict) -> list[str]:
    candidates = [
        row_obj.get("et_path"),
        row_obj.get("pdf_path"),
        row_obj.get("et_paths"),
        row_obj.get("pdf_paths"),
    ]
    if isinstance(payload, dict):
        candidates.extend(
            [
                payload.get("et_path"),
                payload.get("pdf_path"),
                payload.get("et_paths"),
                payload.get("pdf_paths"),
            ]
        )

    out = []
    for cand in candidates:
        out.extend(_as_path_list(cand))

    deduped = []
    seen = set()
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    return deduped


def _resolve_manual_source_jobs(row_obj: dict) -> pd.DataFrame:
    source_rows = as_list(row_obj.get("source_inferred_rows"))
    source_rows = [str(x).strip() for x in source_rows if str(x).strip()]
    if not source_rows:
        return pd.DataFrame()

    source_grouped_path = Path(
        row_obj.get("source_grouped_jobs_path")
        or DEFAULT_GROUPED_JOBS_PATH
    ).expanduser().resolve()
    source_df = load_grouped_jobs(str(source_grouped_path)).copy()
    if "y_row_id" not in source_df.columns:
        raise ValueError(f"Grouped jobs source missing y_row_id: {source_grouped_path}")

    source_df["y_row_id"] = source_df["y_row_id"].astype(str).str.strip()
    matched = source_df[source_df["y_row_id"].isin(source_rows)].copy()
    if matched.empty:
        raise ValueError(
            "Manual Y source_inferred_rows did not match any rows in grouped jobs source: "
            f"{source_grouped_path}"
        )
    return matched


def create_grouped_jobs_from_csv_bridge(
    *,
    selected_modes: list[str],
    y_path_by_mode: dict[str, Path],
    out_path: Path,
) -> pd.DataFrame:
    grouped_frames = []

    for mode in selected_modes:
        if mode not in y_path_by_mode:
            raise KeyError(f"Missing Y path for mode: {mode}")
        y_path = y_path_by_mode[mode]
        cfg = NeedleBridgeConfig(
            matches_index_csv=DEFAULT_MATCHES_INDEX_BY_MODE[mode],
            match_frequencies_csv=None,
            ws_enhanced_csv=DEFAULT_WS_ENHANCED_BY_MODE[mode],
            y_inferred_json=y_path,
            et_path_col="path",
            ws_row_id_col="X1",
            out_row_id_col="row_id",
            y_row_prefix="X1_",
            y_row_pad=4,
            filter_ws_any_needle=True,
            filter_et_any_needle=True,
        )

        out = build_needle_run_plan(cfg)
        run_plan_df = out["run_plan_df"].copy()
        run_plan_df["precedent_mode"] = mode

        grouped_jobs_df = _build_grouped_jobs(run_plan_df)
        grouped_jobs_df["precedent_mode"] = mode
        grouped_jobs_df["y_source_path"] = str(y_path)
        grouped_frames.append(grouped_jobs_df)

    combined_df = pd.concat(grouped_frames, ignore_index=True) if grouped_frames else pd.DataFrame()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_parquet(out_path, index=False)
    return combined_df


def create_grouped_jobs_from_manual_y(
    *,
    y_manual_path: Path,
    out_path: Path,
) -> pd.DataFrame:
    rows_out = []
    missing_paths = []
    source_resolution_errors = {}

    for y_row_id, row_obj in _iter_manual_y_records(y_manual_path):
        payload = row_obj.get("y") if isinstance(row_obj, dict) and isinstance(row_obj.get("y"), dict) else row_obj
        payload = payload if isinstance(payload, dict) else {}

        x_tests = _extract_x_tests(payload)
        et_paths = _extract_et_paths(row_obj, payload)
        source_jobs = pd.DataFrame()

        if not et_paths:
            try:
                source_jobs = _resolve_manual_source_jobs(row_obj)
            except Exception as e:
                source_resolution_errors[y_row_id] = str(e)
                missing_paths.append(y_row_id)
                continue

        row_id = row_obj.get("row_id") or payload.get("row_id") or y_row_id
        precedent_mode = (
            row_obj.get("precedent_mode")
            or row_obj.get("mode")
            or payload.get("precedent_mode")
            or payload.get("mode")
            or "manual"
        )
        match_mode = row_obj.get("match_mode") or payload.get("match_mode") or "MANUAL_Y"
        matched_needles = row_obj.get("matched_needles") or row_obj.get("needles") or payload.get("matched_needles") or []

        if not source_jobs.empty:
            for _, src in source_jobs.iterrows():
                rows_out.append(
                    {
                        "row_id": row_id,
                        "y_row_id": y_row_id,
                        "et_path": str(src.get("et_path")),
                        "precedent_mode": str(src.get("precedent_mode") or precedent_mode),
                        "match_mode": str(src.get("match_mode") or match_mode),
                        "matched_needles": as_list(src.get("matched_needles")),
                        "x_tests": x_tests,
                        "y_source_path": str(y_manual_path),
                    }
                )
        else:
            for et_path in et_paths:
                rows_out.append(
                    {
                        "row_id": row_id,
                        "y_row_id": y_row_id,
                        "et_path": et_path,
                        "precedent_mode": str(precedent_mode),
                        "match_mode": str(match_mode),
                        "matched_needles": as_list(matched_needles),
                        "x_tests": x_tests,
                        "y_source_path": str(y_manual_path),
                    }
                )

    if missing_paths:
        sample = ", ".join(missing_paths[:5])
        sample_err = ""
        if source_resolution_errors:
            first_key = next(iter(source_resolution_errors))
            sample_err = f" Example resolution error for {first_key}: {source_resolution_errors[first_key]}"
        raise ValueError(
            "Manual Y records must include explicit PDF paths or source_inferred_rows that resolve "
            "against the grouped jobs source. "
            f"Missing for {len(missing_paths)} record(s), e.g. {sample}.{sample_err}"
        )

    grouped_df = pd.DataFrame(rows_out)
    if not grouped_df.empty:
        grouped_df = (
            grouped_df
            .groupby(
                ["row_id", "y_row_id", "et_path", "precedent_mode", "match_mode", "y_source_path"],
                dropna=False,
            )
            .agg(
                matched_needles=("matched_needles", lambda s: sorted({n for vals in s for n in as_list(vals)})),
                x_tests=("x_tests", "first"),
            )
            .reset_index()
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_df.to_parquet(out_path, index=False)
    return grouped_df


def apply_workflow_defaults(workflow: str) -> None:
    cfg = WORKFLOW_CONFIG[workflow]
    st.session_state["grouped_jobs_path"] = str(cfg["grouped_path"])
    st.session_state["selected_y_path"] = str(cfg["y_path"])


def _path_status_line(path: Path) -> str:
    mark = "OK" if path.exists() else "Missing"
    return f"- {mark}: `{path}`"


def build_et_matches(mode: str) -> pd.DataFrame:
    _ensure_sys_path()
    import corpus_builder as cb_mod

    importlib.reload(cb_mod)

    needle_pack = _load_needle_pack(mode)
    regex_buckets, regex_folder_map, regex_only_folder_name = _compile_regex_buckets(
        needle_pack.get("regex_buckets", {})
    )
    matches_root = (BASE_MATCHES_ROOT / mode).resolve()

    return cb_mod.run_corpus_builder(
        input_root=ET_INPUT_ROOT,
        matches_root=matches_root,
        needles_all=needle_pack.get("needles_all", []),
        needles_any=needle_pack.get("needles_any", []),
        regex_buckets=regex_buckets,
        regex_folder_map=regex_folder_map,
        cfg_overrides=dict(
            case_sensitive=False,
            min_pages=4,
            text_pages_head=100000,
            text_pages_tail=0,
            max_workers=24,
            submit_chunk_size=2000,
            preserve_structure=True,
            master_csv_name="_matches_index.csv",
            regex_only_folder_name=regex_only_folder_name,
        ),
    )


def build_ws_enrichment(mode: str, model: str = DEFAULT_MODEL) -> pd.DataFrame:
    _ensure_sys_path()
    import y_runner_engine as yr_mod

    importlib.reload(yr_mod)

    spec, tag_spec_path = _compile_tag_spec_from_needles_json(mode)
    allowed_tags, tag_defs = _build_allowed_tags_and_defs_from_spec(spec)
    prompt_fn = _make_needle_prompt_builder(allowed_tags, tag_defs)

    cfg = yr_mod.YRunnerConfig(
        csv_path=DEFAULT_WS_INPUT_CSV,
        text_col=DEFAULT_WS_TEXT_COL,
        out_dir=(REPO_ROOT / "output").resolve(),
        out_enhanced_csv=DEFAULT_WS_ENHANCED_BY_MODE[mode],
        out_y_json=DEFAULT_Y_BY_MODE[mode],
        y_spec_py=DEFAULT_Y_SPEC_PY,
        model=model,
        debug=False,
        debug_slice="3",
        needle_timeout=180,
        y_spec_timeout=180,
        matches_root=(BASE_MATCHES_ROOT / mode).resolve(),
        master_csv_name="_match_frequencies.csv",
        strict_schema_gate=True,
        schema_gate_source="tag_spec",
        tag_spec_path=tag_spec_path,
    )

    df_out, _, _ = yr_mod.run_y_pipeline(
        cfg=cfg,
        allowed_tags=allowed_tags,
        tag_defs=tag_defs,
        needle_prompt_fn=prompt_fn,
        canonicalize_fn=yr_mod.canonicalize_tag,
    )
    return df_out


@st.cache_data(show_spinner=False)
def load_y_rows_by_path(path: str) -> dict:
    return load_y_rows(path)


def make_moltie_modules():
    """
    Import + reload Moltie modules once.
    Returns:
        (RunConfig, AtomQuery, run_agent_on_one_doc, LLMClientConfig, qo_mod, loop_mod, client_mod)
    """
    _ensure_sys_path()

    import moltie.schemas.run_config as rc_mod
    import moltie.llm.verifier_prompt as vp_mod
    import moltie.llm.client as client_mod
    import moltie.schemas.query_object as qo_mod
    import moltie.agent.loop as loop_mod

    importlib.reload(rc_mod)
    importlib.reload(vp_mod)
    importlib.reload(client_mod)
    importlib.reload(qo_mod)
    importlib.reload(loop_mod)

    return (
        rc_mod.RunConfig,
        qo_mod.AtomQuery,
        loop_mod.run_agent_on_one_doc,
        client_mod.LLMClientConfig,
        qo_mod,
        loop_mod,
        client_mod,
    )


@st.cache_resource(show_spinner=False)
def get_paras_cache():
    return {}


def get_paras(pdf_path: Path, loop_mod, paras_cache: dict):
    k = str(pdf_path)
    if k in paras_cache:
        return paras_cache[k]

    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    text = "\n".join([(p.extract_text() or "") for p in reader.pages]).strip()

    paras = [{"para_id": "p00001", "text": text}]
    paras = loop_mod._maybe_rechunk_single_blob_paras(paras)

    paras_cache[k] = paras
    return paras


def make_resume_key(precedent_mode, row_id, y_row_id, et_path, x_key) -> Tuple[str, str, str, str, str]:
    """
    Normalize to strings so resume matching is stable across int/str variations.
    precedent_mode is included to avoid offensive/defensive collisions.
    """
    return (
        "" if precedent_mode is None else str(precedent_mode),
        "" if row_id is None else str(row_id),
        "" if y_row_id is None else str(y_row_id),
        "" if et_path is None else str(et_path),
        "" if x_key is None else str(x_key),
    )


def row_counts_as_completed(row: dict) -> bool:
    """
    A row is considered completed if it reached any terminal write state:
    - verdict present
    - negative_exit present
    - error present
    """
    if not isinstance(row, dict):
        return False

    if row.get("verdict") is not None:
        return True
    if row.get("negative_exit") is not None:
        return True
    if row.get("error") is not None:
        return True

    return False


def load_completed_keys(jsonl_path: Path) -> Set[Tuple[str, str, str, str, str]]:
    """
    Read an existing JSONL and return the set of completed atomic keys:
    (precedent_mode, row_id, y_row_id, et_path, x_key)
    """
    completed = set()

    if not jsonl_path.exists():
        return completed

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            if not row_counts_as_completed(row):
                continue

            key = make_resume_key(
                row.get("precedent_mode"),
                row.get("row_id"),
                row.get("y_row_id"),
                row.get("et_path"),
                row.get("x_key"),
            )

            if all(key):
                completed.add(key)

    return completed


def count_planned_atoms(plan_jobs: pd.DataFrame) -> int:
    total = 0
    if plan_jobs.empty:
        return 0
    for _, job in plan_jobs.iterrows():
        total += len(as_list(job.get("x_tests")))
    return total


def flatten_x_tests_for_preview(plan_jobs: pd.DataFrame, max_rows: int = 300) -> pd.DataFrame:
    rows = []
    for _, job in plan_jobs.iterrows():
        row_id = job.get("row_id")
        y_row_id = job.get("y_row_id")
        precedent_mode = job.get("precedent_mode")
        match_mode = job.get("match_mode")
        et_path = job.get("et_path")
        matched_needles = as_list(job.get("matched_needles"))
        y_source_path = job.get("y_source_path")
        for t in as_list(job.get("x_tests")):
            if not isinstance(t, dict):
                continue
            rows.append(
                {
                    "precedent_mode": precedent_mode,
                    "row_id": row_id,
                    "y_row_id": y_row_id,
                    "match_mode": match_mode,
                    "x_key": t.get("x_key"),
                    "x_name": t.get("x_name", t.get("x_key")),
                    "x_scope": t.get("x_scope", "GENERAL"),
                    "n_matched_needles": len(matched_needles),
                    "et_path": et_path,
                    "y_source_path": y_source_path,
                }
            )
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)


def build_pdf_task_groups(plan_jobs: pd.DataFrame) -> List[dict]:
    """
    Build PDF-first task groups from the selected jobs only.

    Output shape:
    [
        {
            "pdf_path": Path(...),
            "tasks": [
                {
                    "job_i": int,
                    "precedent_mode": str,
                    "row_id": ...,
                    "y_row_id": ...,
                    "match_mode": ...,
                    "matched_needles": [...],
                    "x_key": ...,
                    "x_name": ...,
                },
                ...
            ]
        },
        ...
    ]
    """
    groups = {}

    for job_i, (_, job) in enumerate(plan_jobs.iterrows()):
        row_id = job.get("row_id")
        y_row_id = job.get("y_row_id")
        precedent_mode = job.get("precedent_mode")
        match_mode = job.get("match_mode")
        matched_needles = as_list(job.get("matched_needles"))
        pdf_path = Path(str(job.get("et_path")))
        y_source_path = str(job.get("y_source_path") or "").strip()

        key = str(pdf_path)
        if key not in groups:
            groups[key] = {
                "pdf_path": pdf_path,
                "tasks": [],
            }

        for t in as_list(job.get("x_tests")):
            if not isinstance(t, dict):
                continue

            groups[key]["tasks"].append(
                {
                    "job_i": int(job_i),
                    "precedent_mode": precedent_mode,
                    "row_id": row_id,
                    "y_row_id": y_row_id,
                    "match_mode": match_mode,
                    "matched_needles": matched_needles,
                    "x_key": t.get("x_key"),
                    "x_name": t.get("x_name", t.get("x_key")),
                    "y_source_path": y_source_path,
                }
            )

    return list(groups.values())


def build_pdf_completion_df(
    pdf_groups: List[dict],
    completed_keys: Set[Tuple[str, str, str, str, str]],
) -> pd.DataFrame:
    rows = []
    for group in pdf_groups:
        pdf_path = group["pdf_path"]
        tasks = group["tasks"]
        expected_keys = {
            make_resume_key(
                precedent_mode=task.get("precedent_mode"),
                row_id=task.get("row_id"),
                y_row_id=task.get("y_row_id"),
                et_path=str(pdf_path),
                x_key=task.get("x_key"),
            )
            for task in tasks
        }
        done_keys = {k for k in expected_keys if k in completed_keys}
        rows.append(
            {
                "pdf_path": str(pdf_path),
                "planned_atoms": len(expected_keys),
                "completed_atoms": len(done_keys),
                "remaining_atoms": len(expected_keys - done_keys),
                "status": "complete" if expected_keys and expected_keys <= completed_keys else "incomplete",
            }
        )
    return pd.DataFrame(rows)


# ==========================================================
# Streamlit UI
# ==========================================================
st.set_page_config(page_title="Moltie Runner", layout="wide")
st.title("Moltie Runner")
st.caption(
    "Prepare inputs, build the active corpus, then run Moltie against that corpus."
)

if "grouped_jobs_path" not in st.session_state:
    st.session_state["grouped_jobs_path"] = str(DEFAULT_CSV_GROUPED_JOBS_PATH)

if "selected_y_path" not in st.session_state:
    st.session_state["selected_y_path"] = str(DEFAULT_Y_PATH)

if "active_workflow" not in st.session_state:
    st.session_state["active_workflow"] = "CSV Bridge"

if "workflow_last_applied" not in st.session_state:
    st.session_state["workflow_last_applied"] = st.session_state["active_workflow"]
    apply_workflow_defaults(st.session_state["active_workflow"])

st.subheader("Workflow")

selected_workflow = st.radio(
    "Workflow",
    options=list(WORKFLOW_CONFIG.keys()),
    horizontal=True,
    key="active_workflow",
)

if st.session_state["workflow_last_applied"] != selected_workflow:
    apply_workflow_defaults(selected_workflow)
    st.session_state["workflow_last_applied"] = selected_workflow

workflow_cfg = WORKFLOW_CONFIG[selected_workflow]
active_grouped_path = Path(st.session_state["grouped_jobs_path"])
active_y_source_path = Path(st.session_state["selected_y_path"])
corpus_ready = active_grouped_path.exists()
y_summary_label = "mode-specific from corpus" if selected_workflow == "CSV Bridge" else active_y_source_path.name

summary_a, summary_b, summary_c = st.columns([1.1, 1.2, 1.2])
summary_a.metric("Active Workflow", selected_workflow)
summary_b.metric("Grouped Corpus", active_grouped_path.name)
summary_c.metric("Y Source", y_summary_label)
st.caption(workflow_cfg["caption"])

with st.expander("Active Paths", expanded=False):
    if selected_workflow == "CSV Bridge":
        st.code(
            "\n".join(
                [
                    f"Grouped parquet: {active_grouped_path}",
                    f"Offensive Y: {DEFAULT_Y_BY_MODE['offensive']}",
                    f"Defensive Y: {DEFAULT_Y_BY_MODE['defensive']}",
                ]
            )
        )
    else:
        st.code(
            "\n".join(
                [
                    f"Grouped parquet: {active_grouped_path}",
                    f"Y source: {active_y_source_path}",
                ]
            )
        )

if corpus_ready:
    st.success(
        f"Corpus ready for `{selected_workflow}`. Using `{active_grouped_path.name}`. "
        "No preparation or rebuild is required unless your inputs changed."
    )
else:
    st.warning(
        f"No grouped corpus found for `{selected_workflow}` at `{active_grouped_path}`. "
        "Prepare inputs and create the corpus below."
    )

with st.expander("Prepare Inputs And Create Corpus", expanded=not corpus_ready):
    st.caption("Run only the preparation steps needed for the active workflow.")

    if selected_workflow == "CSV Bridge":
        prep_modes = st.multiselect(
            "Preparation modes",
            options=["offensive", "defensive"],
            default=["offensive", "defensive"],
            help="These control ET harvesting and WS enhancement for the CSV bridge workflow.",
        )

        prep_actions, prep_status = st.columns([1.1, 1.4])

        prep_actions.markdown("**Stage Actions**")
        prep_actions.caption("Run the ET harvester first, then WS enrichment if those files are missing.")

        if prep_actions.button("1. Build ET Matches", key="prep_et_matches", use_container_width=True):
            if not prep_modes:
                st.error("Select at least one preparation mode.")
                st.stop()
            try:
                for mode in prep_modes:
                    df_matches = build_et_matches(mode)
                    st.success(f"ET matches built for {mode}: {DEFAULT_MATCHES_INDEX_BY_MODE[mode]}")
                    st.dataframe(df_matches.head(10), use_container_width=True, height=220)
            except Exception as e:
                st.error(f"Failed to build ET matches: {e}")
                st.stop()

        if prep_actions.button("2. Build WS Enhanced + Y", key="prep_ws_enriched", use_container_width=True):
            if not prep_modes:
                st.error("Select at least one preparation mode.")
                st.stop()
            try:
                for mode in prep_modes:
                    df_ws = build_ws_enrichment(mode, model=DEFAULT_MODEL)
                    st.success(f"WS enriched for {mode}: {DEFAULT_WS_ENHANCED_BY_MODE[mode]}")
                    st.dataframe(df_ws.head(10), use_container_width=True, height=220)
                st.session_state["selected_y_path"] = str(DEFAULT_Y_PATH)
            except Exception as e:
                st.error(f"Failed to build WS enhancement/Y: {e}")
                st.stop()

        prep_status.markdown("**Dependency Status**")
        prep_status.caption(f"ET source root: `{ET_INPUT_ROOT}`")
        prep_status.caption(f"WS source CSV: `{DEFAULT_WS_INPUT_CSV}`")
        for mode in ["offensive", "defensive"]:
            with prep_status.expander(f"{mode.title()} Inputs", expanded=(mode in prep_modes and not corpus_ready)):
                prep_status.caption(_path_status_line(DEFAULT_NEEDLE_JSON_BY_MODE[mode]))
                prep_status.caption(_path_status_line(DEFAULT_MATCHES_INDEX_BY_MODE[mode]))
                prep_status.caption(_path_status_line(DEFAULT_WS_ENHANCED_BY_MODE[mode]))
                prep_status.caption(_path_status_line(BASE_MATCHES_ROOT / mode / "_tag_spec.json"))
    else:
        info_left, info_right = st.columns([1.1, 1.4])
        info_left.markdown("**Manual Workflow**")
        info_left.caption("No ET harvesting or WS enrichment is required here.")
        info_right.markdown("**Dependency Status**")
        info_right.caption(_path_status_line(active_y_source_path))

    st.subheader("Create Corpus")
    st.caption("After input preparation, build the grouped-jobs parquet that the runner will consume.")

    create_left, create_right = st.columns([1, 1])

    if selected_workflow == "CSV Bridge":
        csv_modes = create_left.multiselect(
            "Precedent modes",
            options=["offensive", "defensive"],
            default=["offensive", "defensive"],
            help="Build the grouped corpus from the legacy CSV/needle bridge for the selected modes.",
        )
        csv_y_path_raw = create_left.text_input(
            "Row-derived Y JSON",
            value="mode-specific automatic",
            disabled=True,
            help="CSV Bridge now uses output/Y_inferred_offensive.json and output/Y_inferred_defensive.json automatically.",
        ).strip()
        csv_out_path_raw = create_right.text_input(
            "Output grouped parquet",
            value=str(WORKFLOW_CONFIG["CSV Bridge"]["grouped_path"]),
            help="This parquet will be loaded by the runner below.",
        ).strip()

        with st.expander("Corpus Inputs", expanded=False):
            for mode in ["offensive", "defensive"]:
                st.markdown(f"**{mode}**")
                st.code(str(DEFAULT_MATCHES_INDEX_BY_MODE[mode]))
                st.code(str(DEFAULT_WS_ENHANCED_BY_MODE[mode]))

        if st.button("3. Create Corpus", type="secondary", key="create_corpus_csv", use_container_width=True):
            if not csv_modes:
                st.error("Select at least one precedent mode.")
                st.stop()

            try:
                grouped_df = create_grouped_jobs_from_csv_bridge(
                    selected_modes=csv_modes,
                    y_path_by_mode={mode: DEFAULT_Y_BY_MODE[mode].resolve() for mode in csv_modes},
                    out_path=Path(csv_out_path_raw).expanduser().resolve(),
                )
                st.session_state["grouped_jobs_path"] = str(Path(csv_out_path_raw).expanduser().resolve())
                st.session_state["selected_y_path"] = str(DEFAULT_Y_PATH)
                st.success(f"Corpus created: {csv_out_path_raw}")
                st.dataframe(grouped_df.head(20), use_container_width=True, height=260)
            except Exception as e:
                st.error(f"Failed to create CSV bridge corpus: {e}")
                st.stop()
    else:
        manual_y_path_raw = create_left.text_input(
            "Manual Y JSON",
            value=str(WORKFLOW_CONFIG["Manual JSON"]["y_path"]),
            placeholder="/home/hello/Projects/Statements/output/debug_y/Y_manual.json",
            help="Manual Y records must include PDF path(s) via et_path/pdf_path or et_paths/pdf_paths.",
        ).strip()
        manual_out_path_raw = create_right.text_input(
            "Output grouped parquet",
            value=str(WORKFLOW_CONFIG["Manual JSON"]["grouped_path"]),
            help="This parquet will be loaded by the runner below.",
        ).strip()

        st.caption("Manual JSON bypasses the input CSV. The manual Y file becomes the Y-side source for this corpus.")

        if st.button("2. Create Corpus", type="secondary", key="create_corpus_manual", use_container_width=True):
            try:
                grouped_df = create_grouped_jobs_from_manual_y(
                    y_manual_path=Path(manual_y_path_raw).expanduser().resolve(),
                    out_path=Path(manual_out_path_raw).expanduser().resolve(),
                )
                st.session_state["grouped_jobs_path"] = str(Path(manual_out_path_raw).expanduser().resolve())
                st.session_state["selected_y_path"] = str(Path(manual_y_path_raw).expanduser().resolve())
                st.success(f"Corpus created: {manual_out_path_raw}")
                st.dataframe(grouped_df.head(20), use_container_width=True, height=260)
            except Exception as e:
                st.error(f"Failed to create manual JSON corpus: {e}")
                st.stop()

st.divider()

with st.sidebar:
    st.header("Runner Settings")
    grouped_path = st.text_input(
        "Grouped jobs path (.parquet or .csv)",
        key="grouped_jobs_path",
        help="Active grouped corpus loaded by the runner.",
    )
    y_path = st.text_input(
        "Y source JSON",
        key="selected_y_path",
        help="Fallback Y source for corpora without embedded y_source_path. CSV Bridge corpora use mode-specific Y paths embedded in the parquet.",
    )
    out_dir = Path(
        st.text_input(
            "Output directory",
            str(DEFAULT_OUT_DIR),
        )
    )
    st.caption(f"Workflow: `{selected_workflow}`")

    st.divider()
    st.header("Job Filters")
    DEBUG = st.toggle(
        "Debug mode",
        value=False,
        help="Labels the output file as DEBUG. Execution logic stays the same.",
    )
    ONLY_X_KEY = st.text_input(
        "Only run one X key (optional)",
        value="",
        help="Example: X3",
    ).strip() or None
    MAX_JOBS = st.number_input(
        "Max jobs (optional)",
        min_value=0,
        value=0,
        step=1,
        help="Caps the number of grouped jobs after filtering.",
    )
    MAX_JOBS = int(MAX_JOBS) if MAX_JOBS and int(MAX_JOBS) > 0 else None

    st.divider()
    st.header("Model")
    model_name = st.text_input("Model", value=DEFAULT_MODEL)
    ollama_url = st.text_input("Ollama URL", value=DEFAULT_OLLAMA_URL)
    timeout_s = st.number_input("Timeout (seconds)", min_value=30, value=180, step=30)
    num_predict = st.number_input("Max tokens", min_value=100, value=1000, step=100)
    max_retries = st.number_input("Max retries", min_value=0, value=2, step=1)

    st.divider()
    st.header("Resume")
    RESUME_MODE = st.toggle(
        "Resume from existing JSONL",
        value=False,
        help="Skips atomic tasks already completed in the selected JSONL.",
    )

    default_resume_path = str(
        DEFAULT_OUT_DIR / "batch_results_BATCH_mode_offensive_Y_4_20260306_090256.jsonl"
    )
    resume_jsonl_path_raw = st.text_input(
        "Resume JSONL path",
        value=default_resume_path if RESUME_MODE else "",
        disabled=not RESUME_MODE,
    ).strip()
    resume_jsonl_path = Path(resume_jsonl_path_raw).expanduser() if resume_jsonl_path_raw else None

    WRITE_BACK_TO_SAME_FILE = st.toggle(
        "Append to same resume JSONL",
        value=True,
        disabled=not RESUME_MODE,
        help="If on, missing atoms are appended to the existing JSONL. If off, a new output file is created.",
    )

    st.divider()
    st.caption("Tip: keep Max jobs small until the full pipeline behaves exactly how you want.")

# -------------------------
# Load grouped jobs
# -------------------------
try:
    jobs_df = load_grouped_jobs(grouped_path).copy()
except Exception as e:
    st.error(str(e))
    st.stop()

required_cols = {
    "row_id",
    "y_row_id",
    "et_path",
    "precedent_mode",
    "match_mode",
    "matched_needles",
    "x_tests",
}
missing = required_cols - set(jobs_df.columns)
if missing:
    st.error(f"grouped_jobs_df missing columns: {sorted(missing)}")
    st.stop()

jobs_df["precedent_mode"] = jobs_df["precedent_mode"].astype(str).str.strip().replace("", "unknown")
jobs_df["match_mode"] = jobs_df["match_mode"].astype(str).str.strip().replace("", "unknown")

# -------------------------
# Headline metrics
# -------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Jobs", len(jobs_df))
c2.metric("Unique row_id", jobs_df["row_id"].nunique())
c3.metric("Unique y_row_id", jobs_df["y_row_id"].nunique())
c4.metric("Unique PDFs", jobs_df["et_path"].nunique())
c5.metric("Precedent modes", jobs_df["precedent_mode"].nunique())

st.divider()

# -------------------------
# Filters
# -------------------------
st.subheader("Filter jobs")

f1, f2 = st.columns([1, 1])

all_precedent_modes = sorted(jobs_df["precedent_mode"].dropna().astype(str).unique().tolist())
selected_precedent_modes = f1.multiselect(
    "Precedent mode",
    options=all_precedent_modes,
    default=all_precedent_modes,
    help="Choose offensive, defensive, or both.",
)

all_match_modes = sorted(jobs_df["match_mode"].dropna().astype(str).unique().tolist())
selected_match_modes = f2.multiselect(
    "Match mode",
    options=all_match_modes,
    default=all_match_modes,
    help="Keep NEEDLE_OVERLAP, GENERAL_BROAD_CAPPED, or both.",
)

jobs_view = jobs_df.copy()

if selected_precedent_modes:
    jobs_view = jobs_view[jobs_view["precedent_mode"].isin(selected_precedent_modes)].copy()
else:
    jobs_view = jobs_view.head(0).copy()

if selected_match_modes:
    jobs_view = jobs_view[jobs_view["match_mode"].isin(selected_match_modes)].copy()
else:
    jobs_view = jobs_view.head(0).copy()

# -------------------------
# Select row IDs
# -------------------------
st.subheader("Select WS row_id(s)")

unique_row_ids = sorted(
    pd.Series(jobs_view["row_id"])
    .dropna()
    .unique()
    .tolist()
)

default_rows = unique_row_ids[:1] if unique_row_ids else []
selected_row_ids = st.multiselect(
    "row_id",
    options=unique_row_ids,
    default=default_rows,
    help="These are unique WS row identifiers pulled from grouped jobs.",
)

if selected_row_ids:
    plan_jobs = jobs_view[jobs_view["row_id"].isin(selected_row_ids)].copy()
else:
    plan_jobs = jobs_view.head(0).copy()

# -------------------------
# Optional x_key filter / job cap
# -------------------------
if ONLY_X_KEY is not None and not plan_jobs.empty:
    def _filter_tests(lst):
        return [t for t in as_list(lst) if isinstance(t, dict) and t.get("x_key") == ONLY_X_KEY]

    plan_jobs["x_tests"] = plan_jobs["x_tests"].apply(_filter_tests)
    plan_jobs = plan_jobs[plan_jobs["x_tests"].apply(lambda x: len(as_list(x)) > 0)].copy()

if MAX_JOBS is not None and not plan_jobs.empty:
    plan_jobs = plan_jobs.head(int(MAX_JOBS)).copy()

planned_atom_count = count_planned_atoms(plan_jobs)
selected_modes_for_output = sorted(plan_jobs["precedent_mode"].dropna().astype(str).unique().tolist())

# -------------------------
# Output path / resume
# -------------------------
fresh_out_path = build_output_path(
    out_dir=out_dir,
    selected_row_ids=selected_row_ids,
    selected_precedent_modes=selected_modes_for_output,
    debug=DEBUG,
)

if RESUME_MODE:
    if not resume_jsonl_path_raw:
        st.warning("Resume mode is on, but Resume JSONL path is empty.")
        completed_keys = set()
        out_path = fresh_out_path
    else:
        out_path = resume_jsonl_path if WRITE_BACK_TO_SAME_FILE else fresh_out_path
        try:
            completed_keys = load_completed_keys(resume_jsonl_path) if resume_jsonl_path else set()
        except Exception as e:
            st.error(f"Failed to load resume JSONL: {e}")
            st.stop()
else:
    completed_keys = set()
    out_path = fresh_out_path

all_pdf_groups = build_pdf_task_groups(plan_jobs)

remaining_pdf_groups = []
remaining_atoms = 0

for group in all_pdf_groups:
    pdf_path = group["pdf_path"]
    remaining_tasks = []

    for task in group["tasks"]:
        resume_key = make_resume_key(
            precedent_mode=task.get("precedent_mode"),
            row_id=task.get("row_id"),
            y_row_id=task.get("y_row_id"),
            et_path=str(pdf_path),
            x_key=task.get("x_key"),
        )
        if resume_key not in completed_keys:
            remaining_tasks.append(task)

    if remaining_tasks:
        remaining_pdf_groups.append(
            {
                "pdf_path": pdf_path,
                "tasks": remaining_tasks,
            }
        )
        remaining_atoms += len(remaining_tasks)

planned_pdfs = len(all_pdf_groups)
remaining_pdfs = len(remaining_pdf_groups)
pdf_completion_df = build_pdf_completion_df(all_pdf_groups, completed_keys) if all_pdf_groups else pd.DataFrame()

# -------------------------
# Plan summary
# -------------------------
st.subheader("Plan summary")

s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Selected jobs", len(plan_jobs))
s2.metric("Selected PDFs", plan_jobs["et_path"].nunique() if len(plan_jobs) else 0)
s3.metric("Planned atoms", planned_atom_count)
s4.metric("Remaining atoms", remaining_atoms)
s5.metric("Remaining PDFs", remaining_pdfs)

mode_counts = (
    plan_jobs.groupby("precedent_mode").size().rename("jobs").reset_index()
    if not plan_jobs.empty else pd.DataFrame(columns=["precedent_mode", "jobs"])
)
match_counts = (
    plan_jobs.groupby("match_mode").size().rename("jobs").reset_index()
    if not plan_jobs.empty else pd.DataFrame(columns=["match_mode", "jobs"])
)

c_left, c_right = st.columns(2)
with c_left:
    st.markdown("**Jobs by precedent mode**")
    st.dataframe(mode_counts, use_container_width=True, height=180)
with c_right:
    st.markdown("**Jobs by match mode**")
    st.dataframe(match_counts, use_container_width=True, height=180)

if RESUME_MODE:
    st.info(
        "Resume mode is ON\n\n"
        f"- Resume source: `{resume_jsonl_path}`\n"
        f"- Completed atoms detected: **{len(completed_keys)}**\n"
        f"- Planned atoms in current selection: **{planned_atom_count}**\n"
        f"- Remaining atoms to run: **{remaining_atoms}**\n"
        f"- Remaining PDFs to run: **{remaining_pdfs}**\n"
        f"- Output target: `{out_path}`"
    )
    if not pdf_completion_df.empty:
        st.markdown("**PDF completion**")
        st.dataframe(
            pdf_completion_df.sort_values(["status", "remaining_atoms", "pdf_path"], ascending=[True, False, True]),
            use_container_width=True,
            height=240,
        )
else:
    st.info(f"Output will be written to:\n\n`{out_path}`")

st.caption(f"Active grouped corpus: `{grouped_path}`")
st.caption(f"Active Y source: `{y_path}`")

# -------------------------
# Preview tables
# -------------------------
preview_jobs = plan_jobs[[
    "precedent_mode",
    "row_id",
    "y_row_id",
    "match_mode",
    "et_path",
]].head(200)

st.markdown("**Job preview**")
st.dataframe(preview_jobs, use_container_width=True, height=280)

x_preview = flatten_x_tests_for_preview(plan_jobs, max_rows=300)
if not x_preview.empty:
    st.markdown("**Atomic task preview**")
    st.dataframe(x_preview, use_container_width=True, height=320)

st.divider()

# -------------------------
# Execute
# -------------------------
st.subheader("Execute")
run_clicked = st.button(
    "Run Moltie on selected jobs",
    type="primary",
    disabled=plan_jobs.empty,
)

if run_clicked:
    if RESUME_MODE:
        if resume_jsonl_path is None:
            st.error("Resume mode is on but no resume JSONL path was provided.")
            st.stop()
        if not resume_jsonl_path.exists():
            st.error(f"Resume JSONL not found: {resume_jsonl_path}")
            st.stop()

    if remaining_atoms == 0:
        st.success("Nothing to run. All selected atomic tasks are already completed in the resume file.")
        st.code(str(out_path))
        st.stop()

    (
        RunConfig,
        AtomQuery,
        run_agent_on_one_doc,
        LLMClientConfig,
        qo_mod,
        loop_mod,
        client_mod,
    ) = make_moltie_modules()

    _client_kwargs = dict(
        model=model_name,
        ollama_url=ollama_url,
        timeout_s=int(timeout_s),
        temperature=0.0,
        num_predict=int(num_predict),
        max_retries=int(max_retries),
    )

    try:
        if "stop" in getattr(LLMClientConfig, "__annotations__", {}):
            _client_kwargs["stop"] = []
    except Exception:
        pass

    try:
        if "debug" in getattr(LLMClientConfig, "__annotations__", {}):
            _client_kwargs["debug"] = DEBUG
    except Exception:
        pass

    client_cfg = LLMClientConfig(**_client_kwargs)

    cfg2 = RunConfig.from_dict({
        "debug": DEBUG,
        "harvest_mode": False,
        "max_iters": 2,
        "window_size": 12,
        "stride": 12,
        "top_windows": 1,
        "k_chunks_per_doc": 4,
        "anchors_required": 1,
        "min_hits": 1,
        "thresh_score": 1,
        "thresh_conf": 0.8,
        "plateau_p": 1,
        "eps_improve": 0,
        "iter_temp_enabled": False,
    })

    paras_cache = get_paras_cache()
    rows_cache: dict[str, dict] = {}

    n_ok = 0
    n_neg = 0
    n_err = 0
    n_skip = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if RESUME_MODE and WRITE_BACK_TO_SAME_FILE else "w"

    prog = st.progress(0, text="Running PDFs…")
    status = st.empty()
    live_counts = st.empty()

    total_pdfs = len(remaining_pdf_groups)
    done_pdfs = 0

    with out_path.open(file_mode, encoding="utf-8") as f:
        for group in remaining_pdf_groups:
            pdf_path = group["pdf_path"]
            pdf_tasks = group["tasks"]
            pdf_rows_to_write = []

            prog.progress(
                int((done_pdfs / max(total_pdfs, 1)) * 100),
                text=f"PDF {done_pdfs + 1}/{total_pdfs}",
            )

            status.write(
                f"Running pdf={pdf_path.name} | pending tasks in PDF={len(pdf_tasks)}"
            )

            try:
                if not pdf_path.exists():
                    raise FileNotFoundError(f"PDF missing: {pdf_path}")

                paras = get_paras(pdf_path, loop_mod, paras_cache)
                doc_id = pdf_path.stem

                for task in pdf_tasks:
                    job_i = task["job_i"]
                    precedent_mode = task["precedent_mode"]
                    row_id = task["row_id"]
                    y_row_id = str(task["y_row_id"])
                    match_mode = task["match_mode"]
                    matched_needles = task["matched_needles"]
                    x_key = task["x_key"]
                    x_name = task["x_name"]
                    task_y_source_path = task.get("y_source_path") or y_path

                    resume_key = make_resume_key(
                        precedent_mode=precedent_mode,
                        row_id=row_id,
                        y_row_id=y_row_id,
                        et_path=str(pdf_path),
                        x_key=x_key,
                    )

                    try:
                        cache_key = str(task_y_source_path)
                        rows_for_task = rows_cache.get(cache_key)
                        if rows_for_task is None:
                            rows_for_task = load_y_rows_by_path(task_y_source_path)
                            rows_cache[cache_key] = rows_for_task

                        if y_row_id not in rows_for_task:
                            raise KeyError(f"y_row_id not in Y.rows: {y_row_id}")

                        y_obj = (rows_for_task[y_row_id] or {}).get("y") or {}

                        merged = qo_mod.merge_indicators_and_excludes(y_obj, [x_key])
                        atom = AtomQuery(
                            atom_id=x_key,
                            x_tests=[x_key],
                            proposition=x_name,
                            positive_indicators=merged["positive_indicators"],
                            excludes=merged["excludes"],
                            keyword_seeds=merged["positive_indicators"],
                            expansion_terms=[],
                        )

                        res = run_agent_on_one_doc(doc_id, paras, atom, cfg2, client_cfg)

                        verdict = safe_to_dict(getattr(res, "verdict", None))
                        negative_exit = safe_to_dict(getattr(res, "negative_exit", None))
                        trace_tail = as_list(getattr(res, "trace", None))[-3:]

                        if verdict:
                            n_ok += 1
                        else:
                            n_neg += 1

                        row_out = {
                            "job_i": int(job_i),
                            "precedent_mode": precedent_mode,
                            "row_id": row_id,
                            "y_row_id": y_row_id,
                            "match_mode": match_mode,
                            "matched_needles": matched_needles,
                            "et_path": str(pdf_path),
                            "doc_id": doc_id,
                            "x_key": x_key,
                            "x_name": x_name,
                            "y_source_path": str(task_y_source_path),
                            "verdict": verdict,
                            "negative_exit": negative_exit,
                            "iters": getattr(res, "iters", None),
                            "trace_tail": trace_tail,
                        }
                        pdf_rows_to_write.append(row_out)
                        completed_keys.add(resume_key)

                    except Exception as e_x:
                        n_err += 1
                        err_row = {
                            "job_i": int(job_i),
                            "precedent_mode": precedent_mode,
                            "row_id": row_id,
                            "y_row_id": y_row_id,
                            "match_mode": match_mode,
                            "matched_needles": matched_needles,
                            "et_path": str(pdf_path),
                            "doc_id": doc_id,
                            "x_key": x_key,
                            "x_name": x_name,
                            "y_source_path": str(task_y_source_path),
                            "error": repr(e_x),
                        }
                        pdf_rows_to_write.append(err_row)
                        completed_keys.add(resume_key)

            except Exception as e_pdf:
                # PDF-level failure: write one error row per pending task in that PDF
                for task in pdf_tasks:
                    job_i = task["job_i"]
                    precedent_mode = task["precedent_mode"]
                    row_id = task["row_id"]
                    y_row_id = str(task["y_row_id"])
                    match_mode = task["match_mode"]
                    matched_needles = task["matched_needles"]
                    x_key = task["x_key"]
                    x_name = task["x_name"]
                    task_y_source_path = task.get("y_source_path") or y_path

                    resume_key = make_resume_key(
                        precedent_mode=precedent_mode,
                        row_id=row_id,
                        y_row_id=y_row_id,
                        et_path=str(pdf_path),
                        x_key=x_key,
                    )

                    n_err += 1

                    err_row = {
                        "job_i": int(job_i),
                        "precedent_mode": precedent_mode,
                        "row_id": row_id,
                        "y_row_id": y_row_id,
                        "match_mode": match_mode,
                        "matched_needles": matched_needles,
                        "et_path": str(pdf_path),
                        "doc_id": pdf_path.stem,
                        "x_key": x_key,
                        "x_name": x_name,
                        "y_source_path": str(task_y_source_path),
                        "error": repr(e_pdf),
                    }
                    pdf_rows_to_write.append(err_row)
                    completed_keys.add(resume_key)

            for row in pdf_rows_to_write:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            done_pdfs += 1

            live_counts.write(
                f"ok: {n_ok} | negative: {n_neg} | errors: {n_err} | skipped: {n_skip} | "
                f"pdfs done: {done_pdfs}/{total_pdfs}"
            )

    prog.progress(100, text="Done")
    st.success(
        f"DONE. ok={n_ok} | negative={n_neg} | errors={n_err} | skipped={n_skip}"
    )
    st.code(str(out_path))
