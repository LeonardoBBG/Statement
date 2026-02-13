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
    min_iters_for_anchors: int = 2

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RunConfig":
        # soft parse: missing keys fall back to defaults
        return RunConfig(
            debug=bool(d.get("debug", False)),

            k_candidates=int(d.get("k_candidates", 200)),
            k_verify_docs=int(d.get("k_verify_docs", 30)),
            k_chunks_per_doc=int(d.get("k_chunks_per_doc", 12)),

            window_size=int(d.get("window_size", 24)),
            stride=int(d.get("stride", 12)),
            top_windows=int(d.get("top_windows", 3)),
            min_hits=int(d.get("min_hits", 2)),

            max_iters=int(d.get("max_iters", 3)),
            eps_improve=int(d.get("eps_improve", 3)),
            plateau_p=int(d.get("plateau_p", 2)),

            thresh_score=int(d.get("thresh_score", 70)),
            thresh_conf=int(d.get("thresh_conf", 70)),
            anchors_required=int(d.get("anchors_required", 2)),
            min_iters_for_anchors=int(d.get("min_iters_for_anchors", 2)),

            max_total_verifications=(
                int(d["max_total_verifications"])
                if d.get("max_total_verifications") is not None
                else None
            ),

            y_path=str(d.get("y_path", "")).strip(),
            y_dedup_out=(str(d["y_dedup_out"]).strip() if d.get("y_dedup_out") else None),

            harvest_mode=bool(d.get("harvest_mode", False)),
            harvest_min_score=int(d.get("harvest_min_score", 60)),
            harvest_min_conf=int(d.get("harvest_min_conf", 70)),
            harvest_min_anchors=int(d.get("harvest_min_anchors", 1)),
            memory_max_items=int(d.get("memory_max_items", 30)),
        )
