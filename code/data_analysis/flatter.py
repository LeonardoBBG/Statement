from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def flatten_moltie_jsonl(
    jsonl_path: Path,
    *,
    keep_trace_tail: bool = False,
    max_trace_tail: int | None = 2,
) -> pd.DataFrame:
    """
    Load a Moltie MASTER jsonl and return a single flat dataframe (no merging).
    - Promotes negative_exit.best_attempt when verdict is None
    - Extracts key verdict fields + gating context (match_mode, matched_needles)
    - Adds lightweight retrieval diagnostics from trace_tail (last iter) if present
    - Optionally keeps trace_tail as JSON string (can be large)

    Params
    ------
    jsonl_path : Path
        Path to MASTER_moltie.jsonl
    keep_trace_tail : bool
        If True, store trace_tail as a JSON string column (potentially large).
    max_trace_tail : int | None
        If keep_trace_tail=True, keep only last N trace entries to reduce size.
        If None, keep full trace_tail.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing jsonl: {jsonl_path}")

    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # -------------------------
    # Helpers
    # -------------------------
    def _is_dict(x):
        return isinstance(x, dict)

    def _effective_verdict(row: dict):
        v = row.get("verdict")
        if _is_dict(v):
            return v
        neg = row.get("negative_exit")
        if _is_dict(neg):
            best = neg.get("best_attempt")
            if _is_dict(best):
                return best
        return None

    def _get(d, k, default=None):
        return d.get(k, default) if _is_dict(d) else default

    def _len_list(x):
        return len(x) if isinstance(x, list) else 0

    def _basename(p):
        try:
            return Path(p).name if p else None
        except Exception:
            return None

    # -------------------------
    # Effective verdict
    # -------------------------
    df["effective_verdict"] = df.apply(lambda r: _effective_verdict(r.to_dict()), axis=1)

    # -------------------------
    # Core verdict fields
    # -------------------------
    df["relevant"] = df["effective_verdict"].apply(lambda v: _get(v, "relevant"))
    df["precedent_score"] = df["effective_verdict"].apply(lambda v: _get(v, "precedent_score"))
    df["confidence"] = df["effective_verdict"].apply(lambda v: _get(v, "confidence"))
    df["use_mode"] = df["effective_verdict"].apply(lambda v: _get(v, "use_mode"))
    df["proposition_winner"] = df["effective_verdict"].apply(lambda v: _get(v, "proposition_winner"))
    df["appeal_outcome"] = df["effective_verdict"].apply(lambda v: _get(v, "appeal_outcome"))
    df["successful_party"] = df["effective_verdict"].apply(lambda v: _get(v, "successful_party"))
    df["note"] = df["effective_verdict"].apply(lambda v: _get(v, "note"))

    # Anchors / matched_X
    df["anchors"] = df["effective_verdict"].apply(lambda v: _get(v, "anchors", []))
    df["anchor_count"] = df["anchors"].apply(_len_list)
    df["matched_X"] = df["effective_verdict"].apply(lambda v: _get(v, "matched_X", []))
    df["matched_X_count"] = df["matched_X"].apply(_len_list)

    # -------------------------
    # Negative exit summary
    # -------------------------
    if "negative_exit" in df.columns:
        df["negative_reason"] = df["negative_exit"].apply(lambda v: v.get("reason") if _is_dict(v) else None)
        df["negative_note"] = df["negative_exit"].apply(lambda v: v.get("note") if _is_dict(v) else None)
        df["iters"] = df["negative_exit"].apply(lambda v: v.get("iters") if _is_dict(v) else None)
    else:
        df["negative_reason"] = None
        df["negative_note"] = None
        df["iters"] = None

    # -------------------------
    # Gating context (why job existed)
    # -------------------------
    df["matched_needles"] = df.get("matched_needles")
    df["matched_needles_count"] = df["matched_needles"].apply(_len_list) if "matched_needles" in df.columns else 0
    df["match_mode"] = df.get("match_mode")
    df["job_i"] = df.get("job_i") if "job_i" in df.columns else df.get("i")
    df["_source_file"] = df.get("_source_file")

    # -------------------------
    # Doc metadata
    # -------------------------
    df["et_filename"] = df["et_path"].apply(_basename) if "et_path" in df.columns else None
    if "doc_id" not in df.columns:
        # derive a doc_id from filename (without .pdf) if not present
        df["doc_id"] = df["et_filename"].apply(lambda n: n[:-4] if isinstance(n, str) and n.lower().endswith(".pdf") else n)

    # -------------------------
    # Light retrieval diagnostics (last trace entry)
    # -------------------------
    # Your JSONL often includes trace_tail even when negative_exit exists.
    if "trace_tail" in df.columns:
        def _last_trace(row):
            t = row if isinstance(row, list) and row else None
            return t[-1] if t else None

        df["_last_trace"] = df["trace_tail"].apply(_last_trace)

        # retrieval block
        df["retrieval_method"] = df["_last_trace"].apply(lambda t: _get(_get(t, "retrieval"), "method"))
        df["retrieval_score"] = df["_last_trace"].apply(lambda t: _get(_get(t, "retrieval"), "score"))
        df["retrieval_matched_paras"] = df["_last_trace"].apply(lambda t: _get(_get(t, "retrieval"), "matched_paras"))
        df["retrieval_top_windows"] = df["_last_trace"].apply(lambda t: _get(_get(t, "retrieval"), "top_windows"))

        # window picking (first picked window only, keep it simple)
        def _picked_window_start(t):
            r = _get(t, "retrieval")
            picked = _get(r, "picked_windows", [])
            if isinstance(picked, list) and picked:
                return picked[0].get("start")
            return None

        def _picked_window_end(t):
            r = _get(t, "retrieval")
            picked = _get(r, "picked_windows", [])
            if isinstance(picked, list) and picked:
                return picked[0].get("end")
            return None

        df["picked_window_start"] = df["_last_trace"].apply(_picked_window_start)
        df["picked_window_end"] = df["_last_trace"].apply(_picked_window_end)

        if keep_trace_tail:
            def _trace_to_str(t):
                if not isinstance(t, list):
                    return None
                if max_trace_tail is None:
                    slim = t
                else:
                    slim = t[-int(max_trace_tail):]
                return json.dumps(slim, ensure_ascii=False)
            df["trace_tail_json"] = df["trace_tail"].apply(_trace_to_str)
        else:
            df["trace_tail_json"] = None
    else:
        df["_last_trace"] = None
        df["retrieval_method"] = None
        df["retrieval_score"] = None
        df["retrieval_matched_paras"] = None
        df["retrieval_top_windows"] = None
        df["picked_window_start"] = None
        df["picked_window_end"] = None
        df["trace_tail_json"] = None

    # -------------------------
    # Final column selection (pure flat)
    # -------------------------
    final_cols = [
        # identifiers
        "job_i",
        "_source_file",
        "row_id",
        "y_row_id",
        "x_key",
        "x_name",
        "x_scope" if "x_scope" in df.columns else None,
        # doc
        "et_path",
        "et_filename",
        "doc_id",
        # gating
        "match_mode",
        "matched_needles",
        "matched_needles_count",
        # verdict summary
        "relevant",
        "precedent_score",
        "confidence",
        "anchor_count",
        "matched_X_count",
        "use_mode",
        "proposition_winner",
        "appeal_outcome",
        "successful_party",
        "note",
        # negative exit
        "negative_reason",
        "negative_note",
        "iters",
        # retrieval diagnostics
        "retrieval_method",
        "retrieval_score",
        "retrieval_matched_paras",
        "retrieval_top_windows",
        "picked_window_start",
        "picked_window_end",
        # optional heavy
        "trace_tail_json" if keep_trace_tail else None,
    ]
    final_cols = [c for c in final_cols if c and c in df.columns]

    df_flat = df[final_cols].copy()

    # normalize obvious types
    if "relevant" in df_flat.columns:
        # keep as bool/None; don't force fill
        pass
    for c in ["precedent_score", "confidence", "retrieval_score"]:
        if c in df_flat.columns:
            df_flat[c] = pd.to_numeric(df_flat[c], errors="coerce")

    return df_flat


def pdf_hit_frequency_view(df_flat: pd.DataFrame) -> pd.DataFrame:
    """
    Create a PDF-level relevant-only summary view from the flat DF:
    - one row per doc (doc_id + et_filename + et_path)
    - uses ONLY rows where relevant == True
    - counts relevant runs only
    - computes confidence / precedent metrics from relevant rows only
    - pipes together semantic fields across relevant rows:
        matched_needles, use_mode, proposition_winner,
        appeal_outcome, successful_party, note
    - adds a normalized composite strength metric in [0, 1]:
        precedent_strength

    Notes
    -----
    - This is a curated precedent-summary table, not an all-activity table.
    - Non-relevant rows are intentionally excluded from this view.
    - Numeric summary fields are rounded to 0 decimals, except precedent_strength.
    """

    if df_flat is None or df_flat.empty:
        return pd.DataFrame(
            columns=[
                "doc_id", "et_filename", "et_path",
                "n_relevant_runs",
                "max_anchor_count",
                "max_confidence", "mean_confidence",
                "max_precedent_score", "mean_precedent_score",
                "precedent_strength",
                "matched_needles_pdf",
                "use_mode_pdf",
                "proposition_winner_pdf",
                "appeal_outcome_pdf",
                "successful_party_pdf",
                "note_pdf",
            ]
        )

    df = df_flat.copy()

    # -------------------------
    # Ensure doc columns exist
    # -------------------------
    if "doc_id" not in df.columns:
        if "et_filename" in df.columns:
            df["doc_id"] = df["et_filename"]
        else:
            df["doc_id"] = df.get("et_path").apply(lambda p: Path(p).name if p else None)

    if "et_filename" not in df.columns:
        if "et_path" in df.columns:
            df["et_filename"] = df["et_path"].apply(lambda p: Path(p).name if p else None)
        else:
            df["et_filename"] = None

    if "et_path" not in df.columns:
        df["et_path"] = None

    # -------------------------
    # Relevant-only filter
    # -------------------------
    if "relevant" not in df.columns:
        return pd.DataFrame(
            columns=[
                "doc_id", "et_filename", "et_path",
                "n_relevant_runs",
                "max_anchor_count",
                "max_confidence", "mean_confidence",
                "max_precedent_score", "mean_precedent_score",
                "precedent_strength",
                "matched_needles_pdf",
                "use_mode_pdf",
                "proposition_winner_pdf",
                "appeal_outcome_pdf",
                "successful_party_pdf",
                "note_pdf",
            ]
        )

    df = df[df["relevant"] == True].copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "doc_id", "et_filename", "et_path",
                "n_relevant_runs",
                "max_anchor_count",
                "max_confidence", "mean_confidence",
                "max_precedent_score", "mean_precedent_score",
                "precedent_strength",
                "matched_needles_pdf",
                "use_mode_pdf",
                "proposition_winner_pdf",
                "appeal_outcome_pdf",
                "successful_party_pdf",
                "note_pdf",
            ]
        )

    # -------------------------
    # Helpers
    # -------------------------
    def _is_nullish(x):
        if x is None:
            return True
        try:
            return pd.isna(x)
        except Exception:
            return False

    def _pipe_unique_series(series: pd.Series) -> str | None:
        """
        Dedup + join scalar values from a series, preserving first appearance order.
        """
        seen = set()
        out = []

        for val in series:
            if _is_nullish(val):
                continue

            s = str(val).strip()
            if not s:
                continue

            if s not in seen:
                seen.add(s)
                out.append(s)

        return " | ".join(out) if out else None

    def _pipe_unique_listlike(series: pd.Series) -> str | None:
        """
        Flatten list-like cells (e.g. matched_needles), dedup, preserve first appearance order.
        Falls back to scalar handling if a cell is not list-like.
        """
        seen = set()
        out = []

        for val in series:
            if _is_nullish(val):
                continue

            if isinstance(val, list):
                items = val
            else:
                items = [val]

            for item in items:
                if _is_nullish(item):
                    continue

                s = str(item).strip()
                if not s:
                    continue

                if s not in seen:
                    seen.add(s)
                    out.append(s)

        return " | ".join(out) if out else None

    # -------------------------
    # Numeric safety
    # -------------------------
    for c in ["anchor_count", "confidence", "precedent_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # -------------------------
    # Group + aggregate
    # -------------------------
    grp_cols = ["doc_id", "et_filename", "et_path"]
    g = df.groupby(grp_cols, dropna=False)

    out = g.agg(
        n_relevant_runs=("doc_id", "size"),
        max_anchor_count=("anchor_count", "max") if "anchor_count" in df.columns else ("doc_id", "size"),
        max_confidence=("confidence", "max") if "confidence" in df.columns else ("doc_id", "size"),
        mean_confidence=("confidence", "mean") if "confidence" in df.columns else ("doc_id", "size"),
        max_precedent_score=("precedent_score", "max") if "precedent_score" in df.columns else ("doc_id", "size"),
        mean_precedent_score=("precedent_score", "mean") if "precedent_score" in df.columns else ("doc_id", "size"),
    ).reset_index()

    # -------------------------
    # Pipe semantic fields
    # -------------------------
    piped = g.apply(
        lambda grp: pd.Series({
            "matched_needles_pdf": _pipe_unique_listlike(grp["matched_needles"]) if "matched_needles" in grp.columns else None,
            "use_mode_pdf": _pipe_unique_series(grp["use_mode"]) if "use_mode" in grp.columns else None,
            "proposition_winner_pdf": _pipe_unique_series(grp["proposition_winner"]) if "proposition_winner" in grp.columns else None,
            "appeal_outcome_pdf": _pipe_unique_series(grp["appeal_outcome"]) if "appeal_outcome" in grp.columns else None,
            "successful_party_pdf": _pipe_unique_series(grp["successful_party"]) if "successful_party" in grp.columns else None,
            "note_pdf": _pipe_unique_series(grp["note"]) if "note" in grp.columns else None,
        })
    ).reset_index()

    out = out.merge(piped, on=grp_cols, how="left")

    # -------------------------
    # Normalized composite metric
    # -------------------------
    # Confidence / precedent are assumed to be on a 0-100 style scale.
    # Counts are normalized relative to the max value in the current output.
    runs_max = out["n_relevant_runs"].max() if "n_relevant_runs" in out.columns else None
    anchors_max = out["max_anchor_count"].max() if "max_anchor_count" in out.columns else None

    out["runs_norm"] = (
        out["n_relevant_runs"] / runs_max
        if runs_max and pd.notna(runs_max) and runs_max > 0
        else 0.0
    )

    out["anchors_norm"] = (
        out["max_anchor_count"] / anchors_max
        if anchors_max and pd.notna(anchors_max) and anchors_max > 0
        else 0.0
    )

    out["conf_norm"] = (
        out["mean_confidence"].fillna(0) / 100.0
        if "mean_confidence" in out.columns
        else 0.0
    ).clip(lower=0.0, upper=1.0)

    out["precedent_norm"] = (
        out["mean_precedent_score"].fillna(0) / 100.0
        if "mean_precedent_score" in out.columns
        else 0.0
    ).clip(lower=0.0, upper=1.0)

    out["precedent_strength"] = (
        0.25 * out["runs_norm"] +
        0.25 * out["anchors_norm"] +
        0.25 * out["conf_norm"] +
        0.25 * out["precedent_norm"]
    ).round(4)

    # -------------------------
    # Round displayed numeric fields to 0 decimals
    # -------------------------
    round_cols = [
        "n_relevant_runs",
        "max_anchor_count",
        "max_confidence",
        "mean_confidence",
        "max_precedent_score",
        "mean_precedent_score",
    ]
    for c in round_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(0)

    # Optional: cast obvious integer-like display cols to nullable Int64
    int_like_cols = [
        "n_relevant_runs",
        "max_anchor_count",
        "max_confidence",
        "mean_confidence",
        "max_precedent_score",
        "mean_precedent_score",
    ]
    for c in int_like_cols:
        if c in out.columns:
            try:
                out[c] = out[c].astype("Int64")
            except Exception:
                pass

    # -------------------------
    # Drop helper norm columns
    # -------------------------
    out = out.drop(
        columns=[c for c in ["runs_norm", "anchors_norm", "conf_norm", "precedent_norm"] if c in out.columns]
    )

    # -------------------------
    # Sort strongest PDFs first
    # -------------------------
    sort_cols = [
        "precedent_strength",
        "n_relevant_runs",
        "max_anchor_count",
        "max_confidence",
        "max_precedent_score",
    ]
    sort_cols = [c for c in sort_cols if c in out.columns]

    out = out.sort_values(
        sort_cols,
        ascending=[False] * len(sort_cols),
        na_position="last"
    ).reset_index(drop=True)

    return out