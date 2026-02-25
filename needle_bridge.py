from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class NeedleBridgeConfig:
    # Inputs
    matches_index_csv: Path                 # per-doc ET table (HAS flags + path)
    match_frequencies_csv: Optional[Path]   # aggregate stats (optional; NOT used for join)
    ws_enhanced_csv: Path
    y_inferred_json: Path

    # Column names
    et_path_col: str = "path"              # in _matches_index.csv
    ws_row_id_col: str = "X1"              # in Leonardo_WS_enhanced.csv (misnamed)
    out_row_id_col: str = "row_id"         # normalized output name

    # Mapping WS row number -> Y row id
    y_row_prefix: str = "X1_"              # Y row ids look like X1_0002, X1_0010, etc.
    y_row_pad: int = 4                     # zero-pad width

    # Needle / flag columns used for selection (ORDER MATTERS, but no longer used for exact signature join)
    needle_cols: Tuple[str, ...] = (
        "has__any_needle",
        "has__upheld",
        "has__assumed_intention_regex",
        "has__verbal_warning",
        "has__appeal_scope_regex",
        "has__predetermination",
        "has__no_contemporaneous_evidence",
    )

    # Filters
    filter_ws_any_needle: bool = False     # RECOMMENDED: allow GENERAL WS rows through
    filter_et_any_needle: bool = True      # keep corpus to "needle-matched" docs

    # Hybrid controls
    max_et_docs_per_general_ws_row: int = 500
    general_scope_allowlist: Tuple[str, ...] = ("GENERAL",)  # only run GENERAL X-tests for GENERAL rows


def build_needle_run_plan(cfg: NeedleBridgeConfig) -> Dict[str, pd.DataFrame]:
    """
    HYBRID run plan builder.

    Semantics:
      - Needle WS rows (active selector needles > 0): match ET docs by ANY-overlap of those needles.
      - General WS rows (active selector needles == 0): match a broad ET subset (has__any_needle==1),
        capped at cfg.max_et_docs_per_general_ws_row per WS row, and only run X-tests whose scope
        is in cfg.general_scope_allowlist (default: GENERAL).

    Notes:
      - match_frequencies_csv is aggregate and NOT used for joining.
      - WS numeric row_id is mapped to y_row_id (X1_0002 etc).
      - y_ok must be True (must have x_tests). y_ok false rows are excluded.
      - All needle columns are normalized to 0/1 to avoid type-mismatch shrinkage.

    Returns dict with:
      - et_df, ws_df, row_x_tests_df
      - ws_et_matches_df (row_id, y_row_id, et_path, matched_needle, match_mode)
      - run_plan_df (row_id, y_row_id, et_path, matched_needle, match_mode, x_key, x_name, x_scope)
      - meta
    """

    # ---------- validate files ----------
    for p in (cfg.matches_index_csv, cfg.ws_enhanced_csv, cfg.y_inferred_json):
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing file: {p}")

    if cfg.match_frequencies_csv is not None and not Path(cfg.match_frequencies_csv).exists():
        raise FileNotFoundError(f"Missing file: {cfg.match_frequencies_csv}")

    # ---------- load ----------
    et_raw = pd.read_csv(cfg.matches_index_csv)
    ws_raw = pd.read_csv(cfg.ws_enhanced_csv)

    # ---------- normalize WS row id ----------
    ws = ws_raw.copy()
    if cfg.out_row_id_col not in ws.columns:
        if cfg.ws_row_id_col not in ws.columns:
            raise KeyError(
                f"WS row id column not found. Expected '{cfg.ws_row_id_col}' "
                f"or already-normalized '{cfg.out_row_id_col}'. "
                f"WS columns: {list(ws.columns)[:50]}"
            )
        ws = ws.rename(columns={cfg.ws_row_id_col: cfg.out_row_id_col})

    # ---------- validate columns ----------
    missing_ws = [c for c in cfg.needle_cols if c not in ws.columns]
    missing_et = [c for c in cfg.needle_cols if c not in et_raw.columns]
    if missing_ws:
        raise KeyError(f"WS missing needle columns: {missing_ws}")
    if missing_et:
        raise KeyError(f"ET matches_index missing needle columns: {missing_et}")
    if cfg.et_path_col not in et_raw.columns:
        raise KeyError(f"ET matches_index missing path column '{cfg.et_path_col}'")

    # ---------- helpers ----------
    def _to_bool01(s: pd.Series) -> pd.Series:
        """
        Normalize boolean-ish inputs to 0/1 int.
        Accepts: True/False, 1/0, 1.0/0.0, "TRUE"/"FALSE", "yes"/"no", "1"/"0".
        Unknowns -> 0 by default (safe for gating).
        """
        if s.dtype == bool:
            return s.astype(int)

        mask_na = s.isna()
        s_str = s.astype(str).str.strip().str.lower()

        mapped = s_str.map(
            {
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
                "yes": 1,
                "no": 0,
            }
        )

        # fallback numeric
        if mapped.isna().any():
            num = pd.to_numeric(s, errors="coerce")
            mapped = mapped.fillna(num)

        mapped[mask_na] = pd.NA
        return mapped.fillna(0).astype(int)

    # ---------- map WS numeric row_id -> Y row id ----------
    ws_row_num = pd.to_numeric(ws[cfg.out_row_id_col], errors="raise").astype(int)
    ws["y_row_id"] = ws_row_num.map(lambda n: f"{cfg.y_row_prefix}{n:0{cfg.y_row_pad}d}")

    # ---------- build normalized ws_df / et_df ----------
    ws_df = ws[[cfg.out_row_id_col, "y_row_id", *cfg.needle_cols]].copy()
    et_df = et_raw[[cfg.et_path_col, *cfg.needle_cols]].copy()

    for c in cfg.needle_cols:
        ws_df[c] = _to_bool01(ws_df[c])
        et_df[c] = _to_bool01(et_df[c])

    # ---------- optional any_needle filters ----------
    if cfg.filter_ws_any_needle:
        ws_df = ws_df[ws_df["has__any_needle"] == 1].copy()
    if cfg.filter_et_any_needle:
        et_df = et_df[et_df["has__any_needle"] == 1].copy()

    # ---------- load Y and extract x_tests (+ scope) ----------
    y_root = json.loads(Path(cfg.y_inferred_json).read_text(encoding="utf-8"))
    rows = y_root.get("rows") or {}
    if not isinstance(rows, dict) or not rows:
        raise KeyError("Y_inferred.json has no 'rows' dict at top level.")

    row_records = []
    y_ok_records = []
    for yid, row_obj in rows.items():
        row_obj = row_obj or {}
        y_ok = bool(row_obj.get("y_ok"))
        y_ok_records.append({"y_row_id": yid, "y_ok": y_ok})

        if not y_ok:
            continue

        y_obj = (row_obj.get("y") or {})
        x_tests = y_obj.get("x_tests") or {}
        if isinstance(x_tests, dict) and x_tests:
            for x_key, x_obj in x_tests.items():
                row_records.append(
                    {
                        "y_row_id": yid,
                        "x_key": x_key,
                        "x_name": (x_obj or {}).get("name", x_key),
                        "x_scope": (x_obj or {}).get("scope", "GENERAL"),
                    }
                )

    y_ok_df = pd.DataFrame(y_ok_records)
    row_x_tests_df = pd.DataFrame(row_records)

    if row_x_tests_df.empty:
        raise RuntimeError("No x_tests found anywhere in Y_inferred.json under rows.*.y.x_tests")

    # Keep only WS rows that actually have y_ok true (otherwise they'd explode to nothing later)
    ws_df = ws_df.merge(y_ok_df, on="y_row_id", how="left")
    ws_df["y_ok"] = ws_df["y_ok"].fillna(False)
    ws_df = ws_df[ws_df["y_ok"] == True].copy()  # noqa: E712
    ws_df = ws_df.drop(columns=["y_ok"])

    # ---------- split WS rows into needle-rows vs general-rows ----------
    selector_needles = [c for c in cfg.needle_cols if c != "has__any_needle"]

    active_selector_counts = (ws_df[selector_needles] == 1).sum(axis=1)
    ws_df = ws_df.copy()
    ws_df["active_selector_needles"] = active_selector_counts

    ws_needle = ws_df[ws_df["active_selector_needles"] > 0].copy()
    ws_general = ws_df[ws_df["active_selector_needles"] == 0].copy()

    # ---------- build ET long (for overlap matching on selector needles) ----------
    et_long = et_df[[cfg.et_path_col, *selector_needles]].copy()
    et_long = et_long.melt(
        id_vars=[cfg.et_path_col],
        value_vars=selector_needles,
        var_name="matched_needle",
        value_name="is_active",
    )
    et_long = et_long[et_long["is_active"] == 1].drop(columns=["is_active"]).copy()

    # ---------- Stage 1: needle rows -> ET docs by overlap on active needles ----------
    ws_et_matches_frames = []

    if not ws_needle.empty:
        ws_long = ws_needle[[cfg.out_row_id_col, "y_row_id", *selector_needles]].copy()
        ws_long = ws_long.melt(
            id_vars=[cfg.out_row_id_col, "y_row_id"],
            value_vars=selector_needles,
            var_name="matched_needle",
            value_name="is_active",
        )
        ws_long = ws_long[ws_long["is_active"] == 1].drop(columns=["is_active"]).copy()

        needle_matches = ws_long.merge(et_long, on="matched_needle", how="inner")
        needle_matches = needle_matches.rename(
            columns={
                cfg.out_row_id_col: "row_id",
                cfg.et_path_col: "et_path",
            }
        ).drop_duplicates()

        needle_matches["match_mode"] = "NEEDLE_OVERLAP"
        ws_et_matches_frames.append(needle_matches)

    # ---------- Stage 2: general rows -> broad ET docs (capped) ----------
    if not ws_general.empty:
        # Broad ET set: all docs currently in et_df (already gated by has__any_needle if enabled)
        broad_et_paths = et_df[[cfg.et_path_col]].rename(columns={cfg.et_path_col: "et_path"}).copy()

        # Cap: take the first N deterministically (stable run plan)
        cap_n = int(cfg.max_et_docs_per_general_ws_row)
        if cap_n > 0 and len(broad_et_paths) > cap_n:
            broad_et_paths = broad_et_paths.iloc[:cap_n].copy()

        general_rows = ws_general[[cfg.out_row_id_col, "y_row_id"]].rename(
            columns={cfg.out_row_id_col: "row_id"}
        )
        # Cross join: general rows x capped ET docs
        general_matches = general_rows.merge(broad_et_paths, how="cross")

        general_matches["matched_needle"] = "GENERAL"
        general_matches["match_mode"] = "GENERAL_BROAD_CAPPED"
        ws_et_matches_frames.append(general_matches)

    if not ws_et_matches_frames:
        raise RuntimeError("No WS->ET matches produced (no needle rows and no general rows).")

    ws_et_matches_df = pd.concat(ws_et_matches_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

    # ---------- expand by x_tests ----------
    # For general rows, restrict x_tests by scope allowlist.
    # For needle rows, allow all x_tests for that row.
    row_x_tests_df = row_x_tests_df.copy()
    row_x_tests_df["x_scope"] = row_x_tests_df["x_scope"].fillna("GENERAL")

    # Join in two passes to enforce the scope gate for GENERAL_BROAD_CAPPED
    needle_part = ws_et_matches_df[ws_et_matches_df["match_mode"] == "NEEDLE_OVERLAP"].copy()
    general_part = ws_et_matches_df[ws_et_matches_df["match_mode"] == "GENERAL_BROAD_CAPPED"].copy()

    run_frames = []

    if not needle_part.empty:
        run_frames.append(
            needle_part.merge(row_x_tests_df, on="y_row_id", how="inner")
        )

    if not general_part.empty:
        allow = set(cfg.general_scope_allowlist)
        general_x = row_x_tests_df[row_x_tests_df["x_scope"].isin(allow)].copy()
        run_frames.append(
            general_part.merge(general_x, on="y_row_id", how="inner")
        )

    run_plan_df = pd.concat(run_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)

    # ---------- final columns ----------
    run_plan_df = run_plan_df[
        ["row_id", "y_row_id", "et_path", "matched_needle", "match_mode", "x_key", "x_name", "x_scope"]
    ]

    # cleanup ws_df output
    ws_df_out = ws_df.drop(columns=["active_selector_needles"]).reset_index(drop=True)

    meta = pd.DataFrame(
        [
            {
                "et_docs": int(len(et_df)),
                "ws_rows_y_ok": int(len(ws_df_out)),
                "ws_rows_needle": int(len(ws_needle)),
                "ws_rows_general": int(len(ws_general)),
                "x_tests_rows_total": int(len(row_x_tests_df)),
                "ws_et_matches_rows": int(len(ws_et_matches_df)),
                "run_plan_rows": int(len(run_plan_df)),
                "filter_ws_any_needle": cfg.filter_ws_any_needle,
                "filter_et_any_needle": cfg.filter_et_any_needle,
                "max_et_docs_per_general_ws_row": int(cfg.max_et_docs_per_general_ws_row),
                "general_scope_allowlist": ",".join(cfg.general_scope_allowlist),
            }
        ]
    )

    return {
        "et_df": et_df.reset_index(drop=True),
        "ws_df": ws_df_out,
        "row_x_tests_df": row_x_tests_df.reset_index(drop=True),
        "ws_et_matches_df": ws_et_matches_df,
        "run_plan_df": run_plan_df,
        "meta": meta,
    }