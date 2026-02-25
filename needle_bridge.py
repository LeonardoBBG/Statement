from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class NeedleBridgeConfig:
    # Inputs
    matches_index_csv: Path                 # <-- per-doc ET table (HAS flags + path)
    match_frequencies_csv: Optional[Path]   # <-- aggregate stats (optional; NOT used for join)
    ws_enhanced_csv: Path
    y_inferred_json: Path

    # Column names
    et_path_col: str = "path"              # in _matches_index.csv
    ws_row_id_col: str = "X1"              # in Leonardo_WS_enhanced.csv (misnamed)
    out_row_id_col: str = "row_id"         # normalized output name

    # Mapping WS row number -> Y row id
    y_row_prefix: str = "X1_"              # Y row ids look like X1_0002, X1_0010, etc.
    y_row_pad: int = 4                     # zero-pad width

    # Needle / flag columns used for exact signature match (ORDER MATTERS)
    needle_cols: Tuple[str, ...] = (
        "has__any_needle",
        "has__upheld",
        "has__assumed_intention_regex",
        "has__verbal_warning",
        "has__appeal_scope_regex",
        "has__predetermination",
        "has__no_contemporaneous_evidence",
    )

    # Optional filters (recommended)
    filter_ws_any_needle: bool = True      # drop WS rows with has__any_needle != True
    filter_et_any_needle: bool = True      # drop ET docs with has__any_needle != True


def build_needle_run_plan(cfg: NeedleBridgeConfig) -> Dict[str, pd.DataFrame]:
    """
    Produces an adaptor between:
      (A) ET corpus (matches_index_csv): per-doc needle flags + PDF paths
      (B) WS enhanced csv: per-row needle flags (plus a row number column misnamed X1)
      (C) Y_inferred.json: per-Y-row x_tests located at rows.<Y_ROW_ID>.y.x_tests

    Key points (based on your actual data):
      - match_frequencies_csv is aggregate (match_type/count/percentage) and is NOT used for joining.
      - WS row ids are numeric (1,2,3...) and must be mapped to Y row ids (e.g. X1_0002).
      - Exact matching is done via a stable needle_signature over cfg.needle_cols.
      - The run plan is expanded by x_tests for each Y row.

    Returns dict with:
      - et_df: ET docs with path + flags + needle_signature
      - ws_df: WS rows with row_id + y_row_id + flags + needle_signature
      - row_x_tests_df: expanded (y_row_id, x_key, x_name)
      - run_plan_df: (row_id, y_row_id, needle_signature, et_path, x_key, x_name)
      - ws_sig_counts / et_sig_counts
      - meta
    """

    # ---------- validate files ----------
    for p in (cfg.matches_index_csv, cfg.ws_enhanced_csv, cfg.y_inferred_json):
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing file: {p}")

    # Optional (not used for join)
    if cfg.match_frequencies_csv is not None and not Path(cfg.match_frequencies_csv).exists():
        raise FileNotFoundError(f"Missing file: {cfg.match_frequencies_csv}")

    # ---------- load ----------
    et = pd.read_csv(cfg.matches_index_csv)
    ws = pd.read_csv(cfg.ws_enhanced_csv)

    # ---------- normalize WS row id ----------
    if cfg.out_row_id_col not in ws.columns:
        if cfg.ws_row_id_col not in ws.columns:
            raise KeyError(
                f"WS row id column not found. Expected '{cfg.ws_row_id_col}' or already-normalized '{cfg.out_row_id_col}'. "
                f"WS columns: {list(ws.columns)[:50]}"
            )
        ws = ws.rename(columns={cfg.ws_row_id_col: cfg.out_row_id_col})

    # ---------- validate needle cols exist ----------
    missing_ws = [c for c in cfg.needle_cols if c not in ws.columns]
    missing_et = [c for c in cfg.needle_cols if c not in et.columns]
    if missing_ws:
        raise KeyError(f"WS missing needle columns: {missing_ws}")
    if missing_et:
        raise KeyError(f"ET matches_index missing needle columns: {missing_et}")

    if cfg.et_path_col not in et.columns:
        raise KeyError(
            f"ET matches_index missing path column '{cfg.et_path_col}'. "
            f"ET columns: {list(et.columns)[:50]}"
        )

    # ---------- map WS numeric row_id -> Y row id (e.g., 10 -> X1_0010) ----------
    # WS row_id can be int, float, or string; enforce strict numeric
    ws_row_num = pd.to_numeric(ws[cfg.out_row_id_col], errors="raise").astype(int)
    y_row_id = ws_row_num.map(lambda n: f"{cfg.y_row_prefix}{n:0{cfg.y_row_pad}d}")

    # ---------- build signatures ----------
    def _sig_frame(df: pd.DataFrame, cols: List[str]) -> pd.Series:
        # canonical string signature; exact match depends on stable ordering
        return df[cols].fillna("NA").astype(str).agg("|".join, axis=1)

    et_df = et[[cfg.et_path_col, *cfg.needle_cols]].copy()
    et_df["needle_signature"] = _sig_frame(et_df, list(cfg.needle_cols))

    ws_df = ws[[cfg.out_row_id_col, *cfg.needle_cols]].copy()
    ws_df["y_row_id"] = y_row_id
    ws_df["needle_signature"] = _sig_frame(ws_df, list(cfg.needle_cols))

    # ---------- optional filtering to avoid the useless all-false signature flood ----------
    if cfg.filter_ws_any_needle:
        ws_df = ws_df[ws_df["has__any_needle"] == True].copy()  # noqa: E712
    if cfg.filter_et_any_needle:
        et_df = et_df[et_df["has__any_needle"] == True].copy()  # noqa: E712

    # ---------- counts ----------
    ws_sig_counts = ws_df["needle_signature"].value_counts().rename_axis("needle_signature").reset_index(name="ws_rows")
    et_sig_counts = et_df["needle_signature"].value_counts().rename_axis("needle_signature").reset_index(name="et_docs")

    # ---------- load Y and extract (y_row_id -> x_tests) ----------
    y_root = json.loads(Path(cfg.y_inferred_json).read_text(encoding="utf-8"))
    rows = y_root.get("rows") or {}
    if not isinstance(rows, dict) or not rows:
        raise KeyError("Y_inferred.json has no 'rows' dict at top level (expected keys: rows/source/version).")

    row_records = []
    for yid, row_obj in rows.items():
        y_obj = (row_obj or {}).get("y") or {}
        x_tests = y_obj.get("x_tests") or {}
        if isinstance(x_tests, dict) and x_tests:
            for x_key, x_obj in x_tests.items():
                row_records.append(
                    {
                        "y_row_id": yid,
                        "x_key": x_key,
                        "x_name": (x_obj or {}).get("name", x_key),
                    }
                )

    row_x_tests_df = pd.DataFrame(row_records)
    if row_x_tests_df.empty:
        raise RuntimeError("No x_tests found anywhere in Y_inferred.json under rows.*.y.x_tests")

    # ---------- exact signature join: WS -> ET docs ----------
    ws_to_et = ws_df.merge(
        et_df[[cfg.et_path_col, "needle_signature"]],
        on="needle_signature",
        how="inner",
    )

    # ---------- expand by x_tests using y_row_id ----------
    run_plan_df = ws_to_et.merge(
        row_x_tests_df,
        on="y_row_id",
        how="inner",
    )

    run_plan_df = run_plan_df.rename(
        columns={
            cfg.out_row_id_col: "row_id",
            cfg.et_path_col: "et_path",
        }
    )

    run_plan_df = run_plan_df[
        ["row_id", "y_row_id", "needle_signature", "et_path", "x_key", "x_name"]
    ].drop_duplicates().reset_index(drop=True)

    return {
        "et_df": et_df.reset_index(drop=True),
        "ws_df": ws_df.reset_index(drop=True),
        "row_x_tests_df": row_x_tests_df.reset_index(drop=True),
        "run_plan_df": run_plan_df,
        "ws_sig_counts": ws_sig_counts,
        "et_sig_counts": et_sig_counts,
        "meta": pd.DataFrame(
            [
                {
                    "et_rows": int(len(et_df)),
                    "ws_rows": int(len(ws_df)),
                    "x_tests_rows": int(len(row_x_tests_df)),
                    "run_plan_rows": int(len(run_plan_df)),
                    "filter_ws_any_needle": cfg.filter_ws_any_needle,
                    "filter_et_any_needle": cfg.filter_et_any_needle,
                    "y_row_prefix": cfg.y_row_prefix,
                    "y_row_pad": cfg.y_row_pad,
                }
            ]
        ),
    }