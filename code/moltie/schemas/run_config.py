from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RunConfig:
    # retrieval
    k_candidates: int = 200
    k_verify_docs: int = 30
    k_chunks_per_doc: int = 12

    # loop control
    max_iters: int = 3
    eps_improve: int = 3           # min score improvement to count as progress
    plateau_p: int = 2             # stagnation window

    # acceptance thresholds
    thresh_score: int = 70
    thresh_conf: int = 75
    anchors_required: int = 2
    min_iters_for_anchors: int = 2

    # budget guards
    max_total_verifications: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RunConfig":
        # soft parse: missing keys fall back to defaults
        return RunConfig(
            k_candidates=int(d.get("k_candidates", 200)),
            k_verify_docs=int(d.get("k_verify_docs", 30)),
            k_chunks_per_doc=int(d.get("k_chunks_per_doc", 12)),
            max_iters=int(d.get("max_iters", 3)),
            eps_improve=int(d.get("eps_improve", 3)),
            plateau_p=int(d.get("plateau_p", 2)),
            thresh_score=int(d.get("thresh_score", 70)),
            thresh_conf=int(d.get("thresh_conf", 75)),
            anchors_required=int(d.get("anchors_required", 2)),
            min_iters_for_anchors=int(d.get("min_iters_for_anchors", 2)),
            max_total_verifications=(int(d["max_total_verifications"]) if d.get("max_total_verifications") is not None else None),
        )
