import sys
import json
import importlib
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pandas as pd
import streamlit as st

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
DEFAULT_GROUPED_JOBS_PATH = REPO_ROOT / "output" / "grouped_jobs_df_all_modes.parquet"
DEFAULT_OUT_DIR = REPO_ROOT / "output" / "moltie_batch"

CODE_ROOT = (REPO_ROOT / "code").resolve()
Y_PATH = REPO_ROOT / "output" / "Y_inferred.json"

DEFAULT_MODEL = "mistral-small3.2:latest"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


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


def _safe_int_list_tag(idxs: List[int], max_items: int = 20) -> str:
    """
    Build a compact tag like 'Y_3' or 'Y_3_7_9'. Truncates if too many.
    """
    if not idxs:
        return "Y_NONE"
    idxs = [int(i) for i in idxs]
    if len(idxs) == 1:
        return f"Y_{idxs[0]}"
    head = idxs[:max_items]
    tail_n = len(idxs) - len(head)
    base = "Y_" + "_".join(str(i) for i in head)
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
def load_y_rows() -> dict:
    assert Y_PATH.exists(), f"Missing Y: {Y_PATH}"
    y_root = json.loads(Y_PATH.read_text(encoding="utf-8"))
    rows = y_root.get("rows") or {}
    assert isinstance(rows, dict) and rows, "Y has no rows"
    return rows


def build_output_path(
    out_dir: Path,
    selected_row_ids: List[int],
    selected_precedent_modes: List[str],
    debug: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    row_tag = _safe_int_list_tag(selected_row_ids)
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
                }
            )

    return list(groups.values())


# ==========================================================
# Streamlit UI
# ==========================================================
st.set_page_config(page_title="Moltie Runner", layout="wide")
st.title("Moltie Runner")
st.caption("Run grouped precedent jobs with offensive / defensive mode awareness, preview controls, and safe resume behavior.")

with st.sidebar:
    st.header("Inputs")
    grouped_path = st.text_input(
        "Grouped jobs path (.parquet or .csv)",
        str(DEFAULT_GROUPED_JOBS_PATH),
        help="Prefer the combined grouped jobs file so offensive and defensive jobs can be filtered in-app.",
    )
    out_dir = Path(
        st.text_input(
            "Output directory",
            str(DEFAULT_OUT_DIR),
        )
    )

    st.divider()
    st.header("Execution")
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
else:
    st.info(f"Output will be written to:\n\n`{out_path}`")

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

    rows = load_y_rows()

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

                    resume_key = make_resume_key(
                        precedent_mode=precedent_mode,
                        row_id=row_id,
                        y_row_id=y_row_id,
                        et_path=str(pdf_path),
                        x_key=x_key,
                    )

                    try:
                        if y_row_id not in rows:
                            raise KeyError(f"y_row_id not in Y.rows: {y_row_id}")

                        y_obj = (rows[y_row_id] or {}).get("y") or {}

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
                            "verdict": verdict,
                            "negative_exit": negative_exit,
                            "iters": getattr(res, "iters", None),
                            "trace_tail": trace_tail,
                        }
                        f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
                        f.flush()

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
                            "error": repr(e_x),
                        }
                        f.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                        f.flush()

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
                        "error": repr(e_pdf),
                    }
                    f.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                    f.flush()

                    completed_keys.add(resume_key)

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