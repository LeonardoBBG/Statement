import sys, json, importlib, time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

# ==========================================================
# MOLTIE RUNNER APP
# - loads grouped_jobs_df from disk
# - lets user select which jobs to run
# - shows output path that changes with selection (e.g., Y_3)
# - optional: executes Moltie and writes JSONL
# ==========================================================

# -------------------------
# Constants / defaults
# -------------------------
REPO_ROOT = Path("/home/hello/Projects/Statements").resolve()
DEFAULT_GROUPED_JOBS_PATH = REPO_ROOT / "output" / "grouped_jobs_df.parquet"  # change if you store elsewhere
DEFAULT_OUT_DIR = REPO_ROOT / "output" / "moltie_batch"

CODE_ROOT = (REPO_ROOT / "code").resolve()
Y_PATH = REPO_ROOT / "output" / "Y_inferred.json"

# -------------------------
# Helpers
# -------------------------
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

def _ensure_sys_path():
    assert CODE_ROOT.exists(), f"Missing CODE_ROOT: {CODE_ROOT}"
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))

@st.cache_data(show_spinner=False)
def load_grouped_jobs(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"grouped_jobs_df not found: {p}")
    if p.suffix.lower() in [".parquet"]:
        return pd.read_parquet(p)
    if p.suffix.lower() in [".csv"]:
        return pd.read_csv(p)
    raise ValueError("Unsupported file type. Use .parquet or .csv")

@st.cache_data(show_spinner=False)
def load_y_rows() -> dict:
    assert Y_PATH.exists(), f"Missing Y: {Y_PATH}"
    y_root = json.loads(Y_PATH.read_text(encoding="utf-8"))
    rows = y_root.get("rows") or {}
    assert isinstance(rows, dict) and rows, "Y has no rows"
    return rows

def build_output_path(out_dir: Path, selected_ilocs: List[int], debug: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _safe_int_list_tag(selected_ilocs)
    ts = time.strftime("%Y%m%d_%H%M%S")
    mode = "DEBUG" if debug else "BATCH"
    return out_dir / f"batch_results_{mode}_{tag}_{ts}.jsonl"

# -------------------------
# Optional execution utilities
# -------------------------
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
    Import + reload Moltie modules once (so edits are picked up).
    Returns: (RunConfig, AtomQuery, run_agent_on_one_doc, LLMClientConfig, qo_mod, loop_mod, client_mod)
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
    # resource cache survives reruns
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

# ==========================================================
# Streamlit UI
# ==========================================================
st.set_page_config(page_title="Moltie Runner (Grouped Jobs)", layout="wide")
st.title("Moltie Runner (Grouped Jobs)")

with st.sidebar:
    st.header("Inputs")
    grouped_path = st.text_input("grouped_jobs_df path (.parquet or .csv)", str(DEFAULT_GROUPED_JOBS_PATH))
    out_dir = Path(st.text_input("Output directory", str(DEFAULT_OUT_DIR)))

    st.divider()
    st.header("Selection mode")
    selection_mode = st.radio("Select jobs by…", ["DataFrame index (iloc)", "row_id", "y_row_id"], index=0)

    st.divider()
    st.header("Run controls")
    DEBUG = st.toggle("DEBUG mode", value=False)
    ONLY_X_KEY = st.text_input("ONLY_X_KEY (optional)", value="").strip() or None
    MAX_JOBS = st.number_input("MAX_JOBS (optional)", min_value=0, value=0, step=1)
    MAX_JOBS = int(MAX_JOBS) if MAX_JOBS and int(MAX_JOBS) > 0 else None

    st.caption("Tip: keep MAX_JOBS small until you trust the pipeline end-to-end.")

# Load grouped jobs
try:
    jobs_df = load_grouped_jobs(grouped_path).copy()
except Exception as e:
    st.error(str(e))
    st.stop()

required_cols = {"row_id", "y_row_id", "et_path", "match_mode", "matched_needles", "x_tests"}
missing = required_cols - set(jobs_df.columns)
if missing:
    st.error(f"grouped_jobs_df missing columns: {sorted(missing)}")
    st.stop()

# Basic diagnostics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jobs (rows)", len(jobs_df))
c2.metric("Unique row_id", jobs_df["row_id"].nunique())
c3.metric("Unique y_row_id", jobs_df["y_row_id"].nunique())
c4.metric("Unique PDFs", jobs_df["et_path"].nunique())

st.divider()

# Selection UI
selected_ilocs: List[int] = []

if selection_mode == "DataFrame index (iloc)":
    st.subheader("Select by DataFrame index (iloc)")
    st.caption("Enter indices like: 3 or 3,7,9 or 0-25")

    raw = st.text_input("Indices", value="3")
    raw = raw.strip()

    def parse_ilocs(s: str) -> List[int]:
        if not s:
            return []
        s = s.replace(" ", "")
        out: List[int] = []
        parts = s.split(",")
        for p in parts:
            if "-" in p:
                a, b = p.split("-", 1)
                a_i, b_i = int(a), int(b)
                lo, hi = min(a_i, b_i), max(a_i, b_i)
                out.extend(list(range(lo, hi + 1)))
            else:
                out.append(int(p))
        # de-dupe preserve order
        seen = set()
        out2 = []
        for i in out:
            if i in seen:
                continue
            seen.add(i)
            out2.append(i)
        return out2

    try:
        selected_ilocs = parse_ilocs(raw)
    except Exception as e:
        st.error(f"Could not parse indices: {e}")
        st.stop()

elif selection_mode == "row_id":
    st.subheader("Select by row_id")
    options = sorted(jobs_df["row_id"].dropna().astype(str).unique().tolist())
    chosen = st.multiselect("row_id", options, default=options[:1] if options else [])
    if chosen:
        selected_ilocs = jobs_df.index[jobs_df["row_id"].astype(str).isin(chosen)].tolist()

else:  # y_row_id
    st.subheader("Select by y_row_id")
    options = sorted(jobs_df["y_row_id"].dropna().astype(str).unique().tolist())
    chosen = st.multiselect("y_row_id", options, default=options[:1] if options else [])
    if chosen:
        selected_ilocs = jobs_df.index[jobs_df["y_row_id"].astype(str).isin(chosen)].tolist()

# Filter plan_jobs
plan_jobs = jobs_df.iloc[selected_ilocs].copy() if selected_ilocs else jobs_df.head(0).copy()

if ONLY_X_KEY is not None and not plan_jobs.empty:
    def _filter_tests(lst):
        return [t for t in (lst or []) if t.get("x_key") == ONLY_X_KEY]
    plan_jobs["x_tests"] = plan_jobs["x_tests"].apply(_filter_tests)
    plan_jobs = plan_jobs[plan_jobs["x_tests"].apply(lambda x: len(x or []) > 0)].copy()

if MAX_JOBS is not None and not plan_jobs.empty:
    plan_jobs = plan_jobs.head(MAX_JOBS).copy()

# Output path preview
out_path = build_output_path(out_dir, selected_ilocs[:], DEBUG)

st.subheader("Plan preview")
st.write(f"Selected jobs: **{len(plan_jobs)}**")
if not plan_jobs.empty:
    st.write(
        f"Unique y_row_id: **{plan_jobs['y_row_id'].nunique()}** | "
        f"Unique PDFs: **{plan_jobs['et_path'].nunique()}**"
    )

st.info(f"Output will be written to:\n\n`{out_path}`")

st.dataframe(
    plan_jobs[["row_id", "y_row_id", "match_mode", "et_path"]].head(200),
    use_container_width=True,
    height=320
)

st.divider()

# Optional: run
st.subheader("Execute (optional)")
run_clicked = st.button("Run Moltie on selected jobs", type="primary", disabled=plan_jobs.empty)

if run_clicked:
    # load Y rows
    rows = load_y_rows()

    # import Moltie modules
    RunConfig, AtomQuery, run_agent_on_one_doc, LLMClientConfig, qo_mod, loop_mod, client_mod = make_moltie_modules()

    # client cfg pinned once
    _client_kwargs = dict(
        model="mistral-small3.2:latest",
        ollama_url="http://localhost:11434/api/generate",
        timeout_s=180,
        temperature=0.0,
        num_predict=1000,
        max_retries=2,
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

    # RunConfig pinned once
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

    out_path.parent.mkdir(parents=True, exist_ok=True)

    prog = st.progress(0, text="Running jobs…")
    status = st.empty()

    total_jobs = len(plan_jobs)
    with out_path.open("w", encoding="utf-8") as f:
        for j, (_, job) in enumerate(plan_jobs.iterrows()):
            prog.progress(int((j / max(total_jobs, 1)) * 100), text=f"Job {j+1}/{total_jobs}")
            y_row_id = str(job["y_row_id"])
            pdf_path = Path(str(job["et_path"]))
            row_id = job.get("row_id")
            match_mode = job.get("match_mode")
            matched_needles = job.get("matched_needles") or []

            try:
                if y_row_id not in rows:
                    raise KeyError(f"y_row_id not in Y.rows: {y_row_id}")
                if not pdf_path.exists():
                    raise FileNotFoundError(f"PDF missing: {pdf_path}")

                y_obj = (rows[y_row_id] or {}).get("y") or {}
                paras = get_paras(pdf_path, loop_mod, paras_cache)
                doc_id = pdf_path.stem

                for t in (job.get("x_tests") or []):
                    x_key = t.get("x_key")
                    x_name = t.get("x_name", x_key)
                    try:
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

                        if verdict:
                            n_ok += 1
                        else:
                            n_neg += 1

                        row_out = {
                            "job_i": int(j),
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
                            "trace_tail": (getattr(res, "trace", None) or [])[-3:],
                        }
                        f.write(json.dumps(row_out, ensure_ascii=False) + "\n")

                    except Exception as e_x:
                        n_err += 1
                        f.write(json.dumps({
                            "job_i": int(j),
                            "row_id": row_id,
                            "y_row_id": y_row_id,
                            "match_mode": match_mode,
                            "et_path": str(pdf_path),
                            "doc_id": doc_id,
                            "x_key": x_key,
                            "error": repr(e_x),
                        }, ensure_ascii=False) + "\n")

            except Exception as e_job:
                n_err += 1
                f.write(json.dumps({
                    "job_i": int(j),
                    "row_id": row_id,
                    "y_row_id": y_row_id,
                    "match_mode": match_mode,
                    "et_path": str(pdf_path),
                    "error": repr(e_job),
                }, ensure_ascii=False) + "\n")

            status.write(f"ok: {n_ok} | negative: {n_neg} | errors: {n_err}")

    prog.progress(100, text="Done")
    st.success(f"DONE. ok={n_ok} | negative={n_neg} | errors={n_err}")
    st.code(str(out_path))