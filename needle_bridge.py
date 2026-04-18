from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class NeedleBridgeConfig:
    # Inputs
    matches_index_csv: Path
    match_frequencies_csv: Optional[Path]
    ws_enhanced_csv: Path
    y_inferred_json: Path

    # Column names
    et_path_col: str = "path"
    ws_row_id_col: str = "X1"
    out_row_id_col: str = "row_id"

    # Mapping WS row number -> Y row id
    y_row_prefix: str = "X1_"
    y_row_pad: int = 4

    # Optional explicit needle columns.
    # If None, auto-detect overlap between WS and ET on has__* columns.
    needle_cols: Optional[Tuple[str, ...]] = None

    # Filters
    filter_ws_any_needle: bool = False
    filter_et_any_needle: bool = True

    # Hybrid controls
    max_et_docs_per_general_ws_row: int = 500
    general_scope_allowlist: Tuple[str, ...] = ("GENERAL",)


def build_needle_run_plan(cfg: NeedleBridgeConfig) -> Dict[str, pd.DataFrame]:
    """
    HYBRID run plan builder.

    Semantics:
      - Needle WS rows (active selector needles > 0): match ET docs by ANY-overlap of shared needles.
      - General WS rows (active selector needles == 0): match a broad ET subset (has__any_needle==1 if enabled),
        capped at cfg.max_et_docs_per_general_ws_row per WS row, and only run X-tests whose scope
        is in cfg.general_scope_allowlist.

    Key behavior:
      - If cfg.needle_cols is None, auto-detect shared has__* columns between WS and ET.
      - If cfg.needle_cols is provided, use the overlap of that list with existing WS/ET columns.
      - Do NOT require exact schema equality across WS and ET.
      - Fail only if there is no usable shared bridge vocabulary.
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

    if cfg.et_path_col not in et_raw.columns:
        raise KeyError(f"ET matches_index missing path column '{cfg.et_path_col}'")

    # ---------- helper ----------
    def _to_bool01(s: pd.Series) -> pd.Series:
        """
        Normalize boolean-ish inputs to 0/1 int.
        Accepts: True/False, 1/0, 1.0/0.0, "TRUE"/"FALSE", "yes"/"no", "1"/"0".
        Unknowns -> 0 by default.
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

        if mapped.isna().any():
            num = pd.to_numeric(s, errors="coerce")
            mapped = mapped.fillna(num)

        mapped[mask_na] = pd.NA
        return mapped.fillna(0).astype(int)

    # ---------- detect usable needle columns ----------
    ws_has_cols = sorted([c for c in ws.columns if c.startswith("has__")])
    et_has_cols = sorted([c for c in et_raw.columns if c.startswith("has__")])

    if cfg.needle_cols is None:
        candidate_cols = sorted(set(ws_has_cols) & set(et_has_cols))
    else:
        requested = list(cfg.needle_cols)
        missing_ws = sorted([c for c in requested if c not in ws.columns])
        missing_et = sorted([c for c in requested if c not in et_raw.columns])

        if missing_ws:
            print(f"[warn] WS missing requested needle columns; ignoring: {missing_ws}")
        if missing_et:
            print(f"[warn] ET missing requested needle columns; ignoring: {missing_et}")

        candidate_cols = [c for c in requested if c in ws.columns and c in et_raw.columns]

    # Prefer has__any_needle first if it exists
    needle_cols: List[str] = []
    if "has__any_needle" in candidate_cols:
        needle_cols.append("has__any_needle")
    needle_cols.extend([c for c in candidate_cols if c != "has__any_needle"])

    if not needle_cols:
        raise KeyError(
            "No overlapping needle columns found between WS and ET. "
            f"WS has__ cols: {ws_has_cols} | ET has__ cols: {et_has_cols}"
        )

    selector_needles = [c for c in needle_cols if c != "has__any_needle"]
    if not selector_needles:
        raise KeyError(
            "Only 'has__any_needle' overlaps between WS and ET; no selector needles available for bridging."
        )

    print(f"[bridge] using needle columns: {needle_cols}")

    # ---------- map WS numeric row_id -> Y row id ----------
    ws_row_num = pd.to_numeric(ws[cfg.out_row_id_col], errors="raise").astype(int)
    ws["y_row_id"] = ws_row_num.map(lambda n: f"{cfg.y_row_prefix}{n:0{cfg.y_row_pad}d}")

    # ---------- build normalized ws_df / et_df ----------
    ws_df = ws[[cfg.out_row_id_col, "y_row_id", *needle_cols]].copy()
    et_df = et_raw[[cfg.et_path_col, *needle_cols]].copy()

    for c in needle_cols:
        ws_df[c] = _to_bool01(ws_df[c])
        et_df[c] = _to_bool01(et_df[c])

    # ---------- optional any_needle filters ----------
    if cfg.filter_ws_any_needle and "has__any_needle" in ws_df.columns:
        ws_df = ws_df[ws_df["has__any_needle"] == 1].copy()

    if cfg.filter_et_any_needle:
        if "has__any_needle" not in et_df.columns:
            raise KeyError("filter_et_any_needle=True but ET does not contain 'has__any_needle'")
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

        y_obj = row_obj.get("y") or {}
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

    # Keep only WS rows that actually have y_ok true
    ws_df = ws_df.merge(y_ok_df, on="y_row_id", how="left")
    ws_df["y_ok"] = ws_df["y_ok"].fillna(False)
    ws_df = ws_df[ws_df["y_ok"] == True].copy()  # noqa: E712
    ws_df = ws_df.drop(columns=["y_ok"])

    # ---------- split WS rows into needle-rows vs general-rows ----------
    active_selector_counts = (ws_df[selector_needles] == 1).sum(axis=1)
    ws_df = ws_df.copy()
    ws_df["active_selector_needles"] = active_selector_counts

    ws_needle = ws_df[ws_df["active_selector_needles"] > 0].copy()
    ws_general = ws_df[ws_df["active_selector_needles"] == 0].copy()

    # ---------- build ET long for overlap matching ----------
    et_long = et_df[[cfg.et_path_col, *selector_needles]].copy()
    et_long = et_long.melt(
        id_vars=[cfg.et_path_col],
        value_vars=selector_needles,
        var_name="matched_needle",
        value_name="is_active",
    )
    et_long = et_long[et_long["is_active"] == 1].drop(columns=["is_active"]).copy()

    ws_et_matches_frames = []

    # ---------- Stage 1: needle rows -> ET docs by overlap ----------
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

    # ---------- Stage 2: general rows -> broad ET docs ----------
    if not ws_general.empty:
        broad_et_paths = et_df[[cfg.et_path_col]].rename(columns={cfg.et_path_col: "et_path"}).copy()

        cap_n = int(cfg.max_et_docs_per_general_ws_row)
        if cap_n > 0 and len(broad_et_paths) > cap_n:
            broad_et_paths = broad_et_paths.iloc[:cap_n].copy()

        general_rows = ws_general[[cfg.out_row_id_col, "y_row_id"]].rename(
            columns={cfg.out_row_id_col: "row_id"}
        )

        general_matches = general_rows.merge(broad_et_paths, how="cross")
        general_matches["matched_needle"] = "GENERAL"
        general_matches["match_mode"] = "GENERAL_BROAD_CAPPED"
        ws_et_matches_frames.append(general_matches)

    if not ws_et_matches_frames:
        raise RuntimeError("No WS->ET matches produced (no needle rows and no general rows).")

    ws_et_matches_df = (
        pd.concat(ws_et_matches_frames, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ---------- expand by x_tests ----------
    row_x_tests_df = row_x_tests_df.copy()
    row_x_tests_df["x_scope"] = row_x_tests_df["x_scope"].fillna("GENERAL")

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

    run_plan_df = (
        pd.concat(run_frames, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    run_plan_df = run_plan_df[
        ["row_id", "y_row_id", "et_path", "matched_needle", "match_mode", "x_key", "x_name", "x_scope"]
    ]

    ws_df_out = ws_df.drop(columns=["active_selector_needles"]).reset_index(drop=True)

    meta = pd.DataFrame(
        [
            {
                "bridge_needle_cols": ",".join(needle_cols),
                "bridge_selector_needles": ",".join(selector_needles),
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