from __future__ import annotations

import ast
import base64
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# CONFIG / DEFAULTS
# =========================================================
DEFAULT_WORKBOOK = Path(
    "/home/hello/Projects/Statements/output/moltie_batch/"
    "batch_results_BATCH_mode_offensive_Y_2_20260412_135409__precedent_analysis.xlsx"
)
DEFAULT_FLAT_CSV = Path(
    "/home/hello/Projects/Statements/output/moltie_batch/"
    "batch_results_BATCH_mode_offensive_Y_2_20260412_135409__flat.csv"
)
DEFAULT_PDF_FREQ_CSV = Path(
    "/home/hello/Projects/Statements/output/moltie_batch/"
    "batch_results_BATCH_mode_offensive_Y_2_20260412_135409__pdf_frequency.csv"
)
DEFAULT_PDF_ROOT = Path("/media/hello/Vault/Tribunals/ET_Cases")

st.set_page_config(
    page_title="MOLTIE Precedent Explorer",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #0b1020 0%, #10182b 100%);
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1550px;
    }
    h1, h2, h3 {
        letter-spacing: -0.02em;
    }
    .app-shell {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        padding: 1rem 1.1rem 1.1rem 1.1rem;
        box-shadow: 0 14px 40px rgba(0,0,0,0.18);
        margin-bottom: 1rem;
    }
    .hero {
        background: linear-gradient(135deg, rgba(59,130,246,0.16), rgba(16,185,129,0.10));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 1.25rem 1.25rem 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .eyebrow {
        color: #93c5fd;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        line-height: 1.05;
        margin: 0.15rem 0 0.45rem 0;
    }
    .hero-sub {
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 0;
    }
    .mini-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        min-height: 92px;
    }
    .mini-kicker {
        color: #94a3b8;
        font-size: 0.77rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .mini-value {
        font-size: 1.55rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }
    .mini-note {
        color: #cbd5e1;
        font-size: 0.88rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .section-sub {
        color: #94a3b8;
        margin-bottom: 0.65rem;
    }
    .badge-good {
        display: inline-block;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        background: rgba(16,185,129,0.18);
        color: #86efac;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-warn {
        display: inline-block;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        background: rgba(245,158,11,0.18);
        color: #fcd34d;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-info {
        display: inline-block;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        background: rgba(59,130,246,0.18);
        color: #93c5fd;
        font-size: 0.78rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# GENERAL HELPERS
# =========================================================
def pretty_int(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def pretty_num(value: float | int | None, ndigits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    try:
        return f"{float(value):,.{ndigits}f}"
    except Exception:
        return str(value)


def choose_first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def first_valid(series: pd.Series) -> Optional[str]:
    s = series.dropna().astype(str)
    s = s[s.str.strip() != ""]
    return s.iloc[0] if not s.empty else None


def parse_token_field(value) -> list[str]:
    if pd.isna(value):
        return []

    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()
        if not text:
            return []

        # list-like string e.g. "['a', 'b']"
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    items = parsed
                else:
                    items = [text]
            except Exception:
                items = [text]
        elif "|" in text:
            items = text.split("|")
        else:
            items = [text]

    out = []
    for item in items:
        token = str(item).strip().strip("'\"")
        if token and token.lower() not in {"nan", "none", "null"}:
            out.append(token)
    return out


@st.cache_data(show_spinner=False)
def read_excel_sheets(path_str: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")
    filtered_rows = pd.read_excel(path, sheet_name="filtered_rows")
    pdf_aggregation = pd.read_excel(path, sheet_name="pdf_aggregation")
    return filtered_rows, pdf_aggregation


@st.cache_data(show_spinner=False)
def read_flat_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Flat CSV not found: {path}")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_pdf_freq_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"PDF frequency CSV not found: {path}")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_pdf_bytes(path_str: str) -> bytes:
    return Path(path_str).read_bytes()


def render_pdf(path: Path, height: int = 880) -> None:
    data = read_pdf_bytes(str(path))
    b64 = base64.b64encode(data).decode("utf-8")
    iframe = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}" type="application/pdf"></iframe>'
    st.markdown(iframe, unsafe_allow_html=True)


# =========================================================
# CORE COMPUTATION
# =========================================================
@st.cache_data(show_spinner=False)
def normalise_filtered_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = ["precedent_score", "confidence", "anchor_count", "matched_X_count"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    if "et_filename" not in out.columns and "et_path" in out.columns:
        out["et_filename"] = out["et_path"].apply(lambda x: Path(str(x)).name if pd.notna(x) and str(x).strip() else None)

    if "x_name" not in out.columns:
        out["x_name"] = None
    if "x_key" not in out.columns:
        out["x_key"] = out["x_name"]

    out["relevant_bool"] = out.get("relevant", False) == True
    out["strong_hit"] = (
        out["relevant_bool"]
        & (out["precedent_score"] >= 50)
        & (out["confidence"] >= 60)
    )
    out["weak_hit"] = out["relevant_bool"] & (~out["strong_hit"])
    out["x_weighted_row"] = out["precedent_score"] * out["matched_X_count"]
    return out


@st.cache_data(show_spinner=False)
def compute_x_diagnostics(filtered_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    m = normalise_filtered_rows(filtered_rows)

    x_metrics_local = (
        m.groupby(["y_row_id", "x_key", "x_name"], dropna=False)
        .agg(
            total_runs=("x_key", "size"),
            n_strong=("strong_hit", "sum"),
            n_weak=("weak_hit", "sum"),
            n_relevant_raw=("relevant_bool", "sum"),
            mean_confidence=("confidence", "mean"),
            max_confidence=("confidence", "max"),
            mean_precedent_score=("precedent_score", "mean"),
            max_precedent_score=("precedent_score", "max"),
            anchor_sum=("anchor_count", "sum"),
            max_anchor_count=("anchor_count", "max"),
            matched_X_sum=("matched_X_count", "sum"),
            n_unique_docs=("doc_id", "nunique"),
        )
        .reset_index()
    )
    x_metrics_local["strong_rate_pct"] = (x_metrics_local["n_strong"] / x_metrics_local["total_runs"] * 100).round(2)
    x_metrics_local["raw_relevant_rate_pct"] = (x_metrics_local["n_relevant_raw"] / x_metrics_local["total_runs"] * 100).round(2)
    x_metrics_local = x_metrics_local.sort_values(
        ["strong_rate_pct", "n_strong", "mean_precedent_score", "max_precedent_score"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    x_metrics_global = (
        m.groupby(["x_key", "x_name"], dropna=False)
        .agg(
            total_runs=("x_key", "size"),
            n_strong=("strong_hit", "sum"),
            n_weak=("weak_hit", "sum"),
            n_relevant_raw=("relevant_bool", "sum"),
            mean_confidence=("confidence", "mean"),
            max_confidence=("confidence", "max"),
            mean_precedent_score=("precedent_score", "mean"),
            max_precedent_score=("precedent_score", "max"),
            anchor_sum=("anchor_count", "sum"),
            max_anchor_count=("anchor_count", "max"),
            matched_X_sum=("matched_X_count", "sum"),
            n_unique_docs=("doc_id", "nunique"),
            n_unique_rows=("y_row_id", "nunique"),
        )
        .reset_index()
    )
    x_metrics_global["strong_rate_pct"] = (x_metrics_global["n_strong"] / x_metrics_global["total_runs"] * 100).round(2)
    x_metrics_global["raw_relevant_rate_pct"] = (x_metrics_global["n_relevant_raw"] / x_metrics_global["total_runs"] * 100).round(2)
    x_metrics_global = x_metrics_global.sort_values(
        ["strong_rate_pct", "n_strong", "mean_precedent_score", "total_runs"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    min_runs_global = 10
    max_rate_pct_global = 10.0
    bad_global = x_metrics_global[
        (x_metrics_global["total_runs"] >= min_runs_global)
        & (x_metrics_global["strong_rate_pct"] <= max_rate_pct_global)
    ].copy()
    conf01 = (bad_global["mean_confidence"].fillna(0) / 100.0).clip(0, 1)
    weak_ratio = (bad_global["n_weak"] / bad_global["total_runs"]).fillna(0)
    bad_global["bad_score"] = (
        np.log1p(bad_global["total_runs"])
        * (1 - bad_global["strong_rate_pct"] / 100.0)
        * (1 + 0.6 * conf01)
        * (1 + 0.5 * weak_ratio)
    )
    bad_global = bad_global.sort_values(
        ["bad_score", "total_runs", "strong_rate_pct"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    good_global = x_metrics_global[x_metrics_global["n_strong"] > 0].copy()
    good_global["good_score"] = (
        np.log1p(good_global["n_strong"])
        * (good_global["strong_rate_pct"] / 100.0)
        * (1 + 0.4 * (good_global["mean_precedent_score"] / 100.0).clip(0, 1))
        * (1 + 0.2 * (good_global["anchor_sum"] / good_global["total_runs"]).fillna(0))
    )
    good_global = good_global.sort_values(
        ["good_score", "n_strong", "strong_rate_pct", "mean_precedent_score"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    return {
        "local": x_metrics_local,
        "global": x_metrics_global,
        "bad": bad_global,
        "good": good_global,
    }


@st.cache_data(show_spinner=False)
def compute_exploratory(filtered_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    m = normalise_filtered_rows(filtered_rows)

    needle_stats = defaultdict(lambda: {"freq": 0, "weight_sum": 0.0, "cases": set()})
    x_stats = defaultdict(lambda: {"freq": 0, "weight_sum": 0.0, "cases": set()})
    needle_x_graph = defaultdict(lambda: defaultdict(float))

    for _, row in m.iterrows():
        strength = float(row.get("precedent_score", 0))
        case_id = str(row.get("doc_id", ""))
        needles = parse_token_field(row.get("matched_needles")) or parse_token_field(row.get("matched_needles_pdf"))
        xs = parse_token_field(row.get("x_name")) or parse_token_field(row.get("x_name_pdf"))

        for needle in needles:
            needle_stats[needle]["freq"] += 1
            needle_stats[needle]["weight_sum"] += strength
            needle_stats[needle]["cases"].add(case_id)

        for x_name in xs:
            x_stats[x_name]["freq"] += 1
            x_stats[x_name]["weight_sum"] += strength
            x_stats[x_name]["cases"].add(case_id)

        if needles and xs:
            for needle in needles:
                for x_name in xs:
                    needle_x_graph[needle][x_name] += strength

    def build_stats_table(stats_dict: dict, label_col: str) -> pd.DataFrame:
        rows = []
        for key, payload in stats_dict.items():
            freq = payload["freq"]
            weight_sum = payload["weight_sum"]
            avg_strength = weight_sum / freq if freq else 0.0
            rows.append(
                {
                    label_col: key,
                    "frequency": freq,
                    "weight_sum": round(weight_sum, 6),
                    "avg_strength": round(avg_strength, 6),
                    "case_coverage": len(payload["cases"]),
                }
            )
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(["weight_sum", "frequency", "avg_strength"], ascending=[False, False, False]).reset_index(drop=True)
            out["weight_share"] = (out["weight_sum"] / out["weight_sum"].sum()).round(6)
            out["freq_share"] = (out["frequency"] / out["frequency"].sum()).round(6)
        return out

    def build_graph_table(graph_dict: dict) -> pd.DataFrame:
        rows = []
        for src, targets in graph_dict.items():
            for tgt, weight in targets.items():
                rows.append({"needle": src, "x_name": tgt, "edge_weight": round(weight, 6)})
        out = pd.DataFrame(rows)
        if not out.empty:
            source_totals = out.groupby("needle")["edge_weight"].sum()
            target_totals = out.groupby("x_name")["edge_weight"].sum()
            out["source_total_weight"] = out["needle"].map(source_totals)
            out["target_total_weight"] = out["x_name"].map(target_totals)
            out["edge_weight_norm"] = (
                out["edge_weight"]
                / np.sqrt(out["source_total_weight"] * out["target_total_weight"])
            ).round(6)
            out = out.sort_values(["edge_weight_norm", "edge_weight"], ascending=[False, False]).reset_index(drop=True)
        return out

    return {
        "needle_table": build_stats_table(needle_stats, "needle"),
        "x_table": build_stats_table(x_stats, "x_name"),
        "needle_x_table": build_graph_table(needle_x_graph),
    }


def dataframe_download(df: pd.DataFrame, name: str, label: str) -> None:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=name, mime="text/csv", use_container_width=True)


def mini_card(title: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class='mini-card'>
          <div class='mini-kicker'>{title}</div>
          <div class='mini-value'>{value}</div>
          <div class='mini-note'>{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_bar(df: pd.DataFrame, label_col: str, value_col: str, top_n: int = 12):
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return None
    chart_df = df[[label_col, value_col]].head(top_n).copy()
    chart_df = chart_df.set_index(label_col)
    return chart_df


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### Inputs")
    workbook_path = st.text_input("Precedent workbook", value=str(DEFAULT_WORKBOOK))
    flat_csv_path = st.text_input("Flat CSV", value=str(DEFAULT_FLAT_CSV))
    pdf_freq_csv_path = st.text_input("PDF frequency CSV (optional)", value=str(DEFAULT_PDF_FREQ_CSV))
    pdf_root = st.text_input("PDF root", value=str(DEFAULT_PDF_ROOT))

    st.markdown("### Review controls")
    reader_height = st.slider("PDF reader height", 550, 1400, 900, 50)
    default_top_n = st.slider("Default ranked case count", 5, 100, 25, 5)

    st.markdown("### Signal thresholds")
    st.caption("Diagnostics use the strong-hit logic baked into the app: relevant=True, precedent_score≥50, confidence≥60.")

# =========================================================
# LOAD DATA
# =========================================================
load_error = None
filtered_rows = pd.DataFrame()
pdf_agg = pd.DataFrame()
flat_df = pd.DataFrame()
pdf_freq_df = pd.DataFrame()

try:
    filtered_rows, pdf_agg = read_excel_sheets(workbook_path)
except Exception as e:
    load_error = f"Workbook load failed: {e}"

try:
    if flat_csv_path.strip():
        flat_df = read_flat_csv(flat_csv_path)
except Exception:
    flat_df = pd.DataFrame()

try:
    if pdf_freq_csv_path.strip():
        pdf_freq_df = read_pdf_freq_csv(pdf_freq_csv_path)
except Exception:
    pdf_freq_df = pd.DataFrame()

if load_error:
    st.error(load_error)
    st.stop()

filtered_rows = normalise_filtered_rows(filtered_rows)
if "et_filename" not in pdf_agg.columns and "et_path" in pdf_agg.columns:
    pdf_agg["et_filename"] = pdf_agg["et_path"].apply(lambda x: Path(str(x)).name if pd.notna(x) and str(x).strip() else None)

sort_col_default = "x_weighted_score" if "x_weighted_score" in pdf_agg.columns else "precedent_avg"
if sort_col_default in pdf_agg.columns:
    pdf_agg = pdf_agg.sort_values(sort_col_default, ascending=False, na_position="last").reset_index(drop=True)

x_diag = compute_x_diagnostics(filtered_rows)
exploratory = compute_exploratory(filtered_rows)

# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class='hero'>
      <div class='eyebrow'>MOLTIE precedent explorer</div>
      <div class='hero-title'>Rank the cases. Read the PDFs. Kill the noisy X surfaces.</div>
      <p class='hero-sub'>A single UX layer for precedent review and pipeline diagnostics, wired to your workbook and flat outputs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    mini_card("Ranked PDFs", pretty_int(len(pdf_agg)), "Cases available in pdf_aggregation")
with c2:
    mini_card("Filtered rows", pretty_int(len(filtered_rows)), "Rows with precedent_score > 0")
with c3:
    good_count = len(x_diag["good"])
    mini_card("Good X surfaces", pretty_int(good_count), "Repeat signal producers")
with c4:
    bad_count = len(x_diag["bad"])
    mini_card("Noise candidates", pretty_int(bad_count), "Likely pruning targets")

# =========================================================
# SECTION 1: RANKED PDF REVIEW
# =========================================================
st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>1) Ranked PDF review</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-sub'>Sorted case review with direct PDF reading, ranking controls, and clean case metadata.</div>",
    unsafe_allow_html=True,
)

controls_left, controls_right = st.columns([1.35, 2.65])
with controls_left:
    sort_candidates = [c for c in ["x_weighted_score", "precedent_avg", "precedent_sum", "x_count_sum", "anchor_sum", "confidence_avg"] if c in pdf_agg.columns]
    sort_col = st.selectbox("Sort ranked cases by", options=sort_candidates, index=sort_candidates.index(sort_col_default) if sort_col_default in sort_candidates else 0)
    show_n = st.slider("How many ranked PDFs to show", 5, min(150, max(5, len(pdf_agg))), min(default_top_n, max(5, len(pdf_agg))), 5)
    ascending = st.toggle("Ascending sort", value=False)

    ranked_cases = pdf_agg.sort_values(sort_col, ascending=ascending, na_position="last").reset_index(drop=True)
    ranked_view = ranked_cases.head(show_n).copy()
    ranked_view.insert(0, "rank", range(1, len(ranked_view) + 1))

    key_cols = [c for c in ["rank", "et_filename", "doc_id", sort_col, "precedent_avg", "x_count_sum", "anchor_sum", "confidence_avg"] if c in ranked_view.columns]
    if not key_cols:
        key_cols = ranked_view.columns.tolist()[:8]

    with st.expander("Ranked cases table", expanded=False):
        st.dataframe(ranked_view[key_cols], use_container_width=True, height=520, hide_index=True)
        dataframe_download(ranked_view, "ranked_cases_view.csv", "Download current ranked view")

with controls_right:
    review_row = st.selectbox(
        "Open ranked case",
        options=ranked_cases.index,
        format_func=lambda idx: f"#{idx + 1} — {ranked_cases.iloc[idx].get('et_filename') or ranked_cases.iloc[idx].get('doc_id', 'unknown case')}",
    )
    selected = ranked_cases.iloc[int(review_row)]

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("X weighted score", pretty_num(selected.get("x_weighted_score"), 2))
    with k2:
        st.metric("Precedent avg", pretty_num(selected.get("precedent_avg"), 2))
    with k3:
        st.metric("X count sum", pretty_num(selected.get("x_count_sum"), 0))
    with k4:
        st.metric("Anchor sum", pretty_num(selected.get("anchor_sum"), 0))


review_left, review_right = st.columns([1.05, 1.95])
with review_left:
    st.markdown("<span class='badge-good'>Review notes</span>", unsafe_allow_html=True)
    file_label = selected.get("et_filename") or selected.get("doc_id") or "Unknown PDF"
    st.markdown(f"**{file_label}**")
    st.caption(selected.get("et_path", "No direct path found in workbook."))

    same_doc_rows = filtered_rows[filtered_rows["doc_id"].astype(str) == str(selected.get("doc_id"))].copy()
    if not same_doc_rows.empty:
        st.write("Top supporting rows for this PDF")
        support_cols = [c for c in ["y_row_id", "x_name", "precedent_score", "confidence", "matched_X_count", "anchor_count", "note", "negative_reason"] if c in same_doc_rows.columns]
        same_doc_rows = same_doc_rows.sort_values(["precedent_score", "matched_X_count", "confidence"], ascending=[False, False, False])
        st.dataframe(same_doc_rows[support_cols].head(15), use_container_width=True, height=360, hide_index=True)
    else:
        st.info("No supporting filtered rows found for this document in filtered_rows.")

with review_right:
    pdf_path_val = selected.get("et_path")
    pdf_path = Path(str(pdf_path_val)) if pd.notna(pdf_path_val) and str(pdf_path_val).strip() else None
    if pdf_path and pdf_path.exists():
        render_pdf(pdf_path, height=reader_height)
    else:
        st.warning("PDF path not available or file not found. The ranked sheet needs a valid et_path for direct reading.")

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# SECTION 2: DIAGNOSTICS
# =========================================================
st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>2) Diagnostics</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-sub'>Global and local X metrics, noise triage, good surfaces, and the exploratory needle/X layer.</div>",
    unsafe_allow_html=True,
)

overview_left, overview_right = st.columns([1.1, 1.9])
with overview_left:
    top_good = x_diag["good"].head(10)
    top_bad = x_diag["bad"].head(10)
    st.markdown("<span class='badge-good'>Fast read</span>", unsafe_allow_html=True)
    st.write(f"Top good surfaces available: **{len(x_diag['good']):,}**")
    st.write(f"Top noise candidates available: **{len(x_diag['bad']):,}**")
    if not exploratory["needle_table"].empty and exploratory["needle_table"]["needle"].nunique() <= 1:
        st.markdown("<span class='badge-warn'>Needle diversity low</span>", unsafe_allow_html=True)
        st.caption("Needle charting is descriptive only here. X surfaces matter more.")

with overview_right:
    col_a, col_b = st.columns(2)
    with col_a:
        chart = top_bar(x_diag["good"], "x_name", "good_score", top_n=12)
        if chart is not None:
            st.caption("Good X surfaces")
            st.bar_chart(chart)
    with col_b:
        chart = top_bar(x_diag["bad"], "x_name", "bad_score", top_n=12)
        if chart is not None:
            st.caption("Noise triage")
            st.bar_chart(chart)


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Global X metrics",
    "Noise triage",
    "Good X surfaces",
    "Local X metrics",
    "Exploratory X",
    "Needle → X",
])

with tab1:
    left, right = st.columns([1, 2])
    with left:
        min_runs = st.slider("Minimum total_runs", 0, int(max(0, x_diag["global"]["total_runs"].max())) if not x_diag["global"].empty else 0, 0, 1)
        min_rate = st.slider("Minimum strong_rate_pct", 0.0, 100.0, 0.0, 1.0)
    global_view = x_diag["global"].copy()
    if not global_view.empty:
        global_view = global_view[(global_view["total_runs"] >= min_runs) & (global_view["strong_rate_pct"] >= min_rate)]
    st.dataframe(global_view.head(200), use_container_width=True, height=520, hide_index=True)
    dataframe_download(global_view, "x_metrics_global_filtered.csv", "Download global X metrics")

with tab2:
    st.caption("High bad_score + many runs + low strong_rate_pct = likely pruning candidates.")
    st.dataframe(x_diag["bad"].head(200), use_container_width=True, height=520, hide_index=True)
    dataframe_download(x_diag["bad"], "x_bad_global.csv", "Download noise triage")

with tab3:
    st.caption("High good_score surfaces are your repeat signal producers.")
    st.dataframe(x_diag["good"].head(200), use_container_width=True, height=520, hide_index=True)
    dataframe_download(x_diag["good"], "x_good_global.csv", "Download good surfaces")

with tab4:
    st.caption("Row-level X diagnostics for QA and drilling into odd WS/X combinations.")
    st.dataframe(x_diag["local"].head(250), use_container_width=True, height=520, hide_index=True)
    dataframe_download(x_diag["local"], "x_metrics_local.csv", "Download local X metrics")

with tab5:
    left, right = st.columns(2)
    with left:
        st.caption("Exploratory X weighted distribution")
        st.dataframe(exploratory["x_table"].head(150), use_container_width=True, height=460, hide_index=True)
        dataframe_download(exploratory["x_table"], "exploratory_x_table.csv", "Download X exploratory table")
    with right:
        chart = top_bar(exploratory["x_table"], "x_name", "weight_sum", top_n=15)
        if chart is not None:
            st.bar_chart(chart)

with tab6:
    left, right = st.columns(2)
    with left:
        if not exploratory["needle_table"].empty:
            st.caption("Needle distribution")
            st.dataframe(exploratory["needle_table"].head(100), use_container_width=True, height=220, hide_index=True)
        else:
            st.info("No valid needle data found after parsing the filtered rows.")

        st.caption("Needle → X weighted co-occurrence")
        st.dataframe(exploratory["needle_x_table"].head(150), use_container_width=True, height=290, hide_index=True)
        dataframe_download(exploratory["needle_x_table"], "needle_x_edges.csv", "Download needle → X edges")
    with right:
        if not exploratory["needle_table"].empty and exploratory["needle_table"]["needle"].nunique() > 1:
            chart = top_bar(exploratory["needle_table"], "needle", "weight_sum", top_n=12)
            if chart is not None:
                st.caption("Needle weighted distribution")
                st.bar_chart(chart)
        else:
            st.info("Needle diversity is too low to justify serious charting. Focus on X distribution and edges.")

        chart = top_bar(exploratory["needle_x_table"], "x_name", "edge_weight", top_n=12)
        if chart is not None:
            st.caption("Top edge targets by raw weighted mass")
            st.bar_chart(chart)

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.caption(
    "Defaults are wired to your current MOLTIE run. The app reads the workbook for ranked PDF review and computes diagnostics directly from filtered_rows, so it stays aligned with the upstream notebook logic."
)