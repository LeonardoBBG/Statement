from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Literal, Optional

from .verdict import Verdict, VerdictValidator


NegReason = Literal["exhausted", "plateau", "no_anchors", "mismatch", "budget"]


@dataclass(frozen=True)
class NegativeExit:
    """
    A structured 'nothing to see here' outcome for one atom query.
    """
    atom_id: str
    reason: NegReason
    best_attempt: Optional[Dict[str, Any]] = None   # store Verdict dict (or None)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NegativeExit":
        NegativeExitValidator.validate(d)
        return NegativeExit(
            atom_id=str(d["atom_id"]).strip(),
            reason=d["reason"],
            best_attempt=d.get("best_attempt"),
            note=str(d.get("note", "")).strip(),
        )

    @staticmethod
    def from_best(atom_id: str, reason: NegReason, best: Optional[Verdict], note: str = "") -> "NegativeExit":
        return NegativeExit(
            atom_id=atom_id,
            reason=reason,
            best_attempt=(best.to_dict() if best is not None else None),
            note=note.strip(),
        )


class NegativeExitValidator:
    REASONS = {"exhausted", "plateau", "no_anchors", "mismatch", "budget"}

    @staticmethod
    def validate(d: Dict[str, Any]) -> None:
        if not isinstance(d, dict):
            raise TypeError("NegativeExit must be a dict")
        for k in ("atom_id", "reason"):
            if k not in d:
                raise ValueError(f"NegativeExit missing required field: {k}")

        if not str(d["atom_id"]).strip():
            raise ValueError("NegativeExit.atom_id must be non-empty")

        r = d["reason"]
        if r not in NegativeExitValidator.REASONS:
            raise ValueError(f"NegativeExit.reason invalid: {r}")

        ba = d.get("best_attempt")
        if ba is not None:
            if not isinstance(ba, dict):
                raise ValueError("NegativeExit.best_attempt must be a dict if present")
            # ensure it's at least a valid Verdict shape
            VerdictValidator.validate(ba)
