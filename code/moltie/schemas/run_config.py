from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RunConfig:
    # ----------------
    # global behavior
    # ----------------
    debug: bool = False

    # retrieval
    k_candidates: int = 200
    k_verify_docs: int = 30
    k_chunks_per_doc: int = 12

    # windowed retrieval params (coverage)
    window_size: int = 24
    stride: int = 12
    top_windows: int = 3
    min_hits: int = 2

    # loop control
    max_iters: int = 3
    eps_improve: int = 3           # min score improvement to count as progress
    plateau_p: int = 2             # stagnation window

    # acceptance thresholds
    thresh_score: int = 70
    thresh_conf: int = 70
    anchors_required: int = 2

    # budget guards
    max_total_verifications: Optional[int] = None

    # NEW: Y plumbing
    y_path: str = ""                          # path to inferred Y json (required in practice)
    y_dedup_out: Optional[str] = None         # where to save deduped-Y view (optional)

    # --- harvest mode (junior sweep) ---
    harvest_mode: bool = False
    harvest_min_score: int = 60
    harvest_min_conf: int = 70
    harvest_min_anchors: int = 1

    memory_max_items: int = 30

    # -------------------------------
    # NEW: per-iteration temperature
    # -------------------------------
    # If enabled, loop.py will override llm_cfg.temperature per-iteration.
    iter_temp_enabled: bool = False
    iter_temp_start: float = 0.0     # temp at iter=1
    iter_temp_end: float = 0.0       # temp at iter=max_iters
    iter_temp_curve: str = "linear"  # "linear" | "exp"
    iter_temp_exp_k: float = 2.0     # curve shaping for exp
    iter_temp_cap: float = 0.8       # safety cap

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RunConfig":
        iter_temp_curve = str(d.get("iter_temp_curve", "linear")).strip().lower() or "linear"
        if iter_temp_curve not in {"linear", "exp"}:
            raise ValueError("RunConfig.iter_temp_curve must be 'linear' or 'exp'")

        window_size = max(1, int(d.get("window_size", 24)))
        stride = max(1, int(d.get("stride", 12)))
        top_windows = max(1, int(d.get("top_windows", 3)))
        min_hits = max(0, int(d.get("min_hits", 2)))
        max_iters = max(1, int(d.get("max_iters", 3)))
        eps_improve = max(0, int(d.get("eps_improve", 3)))
        plateau_p = max(1, int(d.get("plateau_p", 2)))
        thresh_score = max(0, int(d.get("thresh_score", 70)))
        thresh_conf = max(0, float(d.get("thresh_conf", 70)))
        anchors_required = max(0, int(d.get("anchors_required", 2)))
        k_chunks_per_doc = max(1, int(d.get("k_chunks_per_doc", 12)))
        harvest_min_score = max(0, int(d.get("harvest_min_score", 60)))
        harvest_min_conf = max(0, int(d.get("harvest_min_conf", 70)))
        harvest_min_anchors = max(0, int(d.get("harvest_min_anchors", 1)))
        memory_max_items = max(1, int(d.get("memory_max_items", 30)))
        iter_temp_start = max(0.0, float(d.get("iter_temp_start", 0.0)))
        iter_temp_end = max(0.0, float(d.get("iter_temp_end", 0.0)))
        iter_temp_exp_k = max(0.1, float(d.get("iter_temp_exp_k", 2.0)))
        iter_temp_cap = max(0.0, float(d.get("iter_temp_cap", 0.8)))

        # soft parse: missing keys fall back to defaults
        return RunConfig(
            debug=bool(d.get("debug", False)),

            k_candidates=int(d.get("k_candidates", 200)),
            k_verify_docs=int(d.get("k_verify_docs", 30)),
            k_chunks_per_doc=k_chunks_per_doc,

            window_size=window_size,
            stride=stride,
            top_windows=top_windows,
            min_hits=min_hits,

            max_iters=max_iters,
            eps_improve=eps_improve,
            plateau_p=plateau_p,

            thresh_score=thresh_score,
            thresh_conf=thresh_conf,
            anchors_required=anchors_required,

            max_total_verifications=(
                int(d["max_total_verifications"])
                if d.get("max_total_verifications") is not None
                else None
            ),

            y_path=str(d.get("y_path", "")).strip(),
            y_dedup_out=(str(d["y_dedup_out"]).strip() if d.get("y_dedup_out") else None),

            harvest_mode=bool(d.get("harvest_mode", False)),
            harvest_min_score=harvest_min_score,
            harvest_min_conf=harvest_min_conf,
            harvest_min_anchors=harvest_min_anchors,
            memory_max_items=memory_max_items,

            # per-iter temperature (safe defaults)
            iter_temp_enabled=bool(d.get("iter_temp_enabled", False)),
            iter_temp_start=iter_temp_start,
            iter_temp_end=iter_temp_end,
            iter_temp_curve=iter_temp_curve,
            iter_temp_exp_k=iter_temp_exp_k,
            iter_temp_cap=iter_temp_cap,
        )
