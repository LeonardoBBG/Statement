from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional


UseMode = Literal["support", "contrast", "harmful"]
Party = Literal["claimant", "respondent", "mixed", "unclear"]
AppealOutcome = Literal["allowed", "dismissed", "remitted", "mixed", "unknown"]


@dataclass(frozen=True)
class Anchor:
    """
    Verbatim quote anchor extracted from the FULL appeal doc.
    para_id should be stable (e.g., "p123" or "¶123" or "chunk_17:p3").
    """
    para_id: str
    quote: str
    why_it_matters: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Anchor":
        AnchorValidator.validate(d)
        return Anchor(
            para_id=str(d["para_id"]).strip(),
            quote=str(d["quote"]).strip(),
            why_it_matters=str(d["why_it_matters"]).strip(),
        )


@dataclass(frozen=True)
class Verdict:
    """
    Output of verify_with_llm() for one doc vs one atom query.
    """
    atom_id: str
    doc_id: str

    relevant: bool
    matched_X: List[str] = field(default_factory=list)

    precedent_score: int = 0            # 0-100 (objective-ish)
    confidence: int = 0                 # 0-100 (model confidence, but constrained by rules)
    anchors: List[Anchor] = field(default_factory=list)

    use_mode: UseMode = "contrast"
    proposition_winner: Party = "unclear"

    # optional metadata
    appeal_outcome: AppealOutcome = "unknown"
    successful_party: Party = "unclear"

    distinguishers: List[str] = field(default_factory=list)
    note: str = ""

    # retrieval trace (optional but helpful)
    retrieval_score: Optional[float] = None
    retrieval_method: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["anchors"] = [a.to_dict() for a in self.anchors]
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Verdict":
        VerdictValidator.validate(d)
        return Verdict(
            atom_id=str(d["atom_id"]).strip(),
            doc_id=str(d["doc_id"]).strip(),
            relevant=bool(d["relevant"]),
            matched_X=list(d.get("matched_X", [])),
            precedent_score=int(d.get("precedent_score", 0)),
            confidence=int(d.get("confidence", 0)),
            anchors=[Anchor.from_dict(x) for x in (d.get("anchors") or [])],
            use_mode=d.get("use_mode", "contrast"),
            proposition_winner=d.get("proposition_winner", "unclear"),
            appeal_outcome=d.get("appeal_outcome", "unknown"),
            successful_party=d.get("successful_party", "unclear"),
            distinguishers=list(d.get("distinguishers", [])),
            note=str(d.get("note", "")).strip(),
            retrieval_score=(float(d["retrieval_score"]) if d.get("retrieval_score") is not None else None),
            retrieval_method=(str(d["retrieval_method"]).strip() if d.get("retrieval_method") else None),
        )


# ----------------------------
# Validators
# ----------------------------

class AnchorValidator:
    @staticmethod
    def validate(d: Dict[str, Any]) -> None:
        if not isinstance(d, dict):
            raise TypeError("Anchor must be a dict")
        for k in ("para_id", "quote", "why_it_matters"):
            if k not in d:
                raise ValueError(f"Anchor missing required field: {k}")

        pid = str(d["para_id"]).strip()
        if not pid:
            raise ValueError("Anchor.para_id must be non-empty")

        quote = str(d["quote"]).strip()
        if len(quote) < 10:
            raise ValueError("Anchor.quote too short")

        why = str(d["why_it_matters"]).strip()
        if len(why) < 5:
            raise ValueError("Anchor.why_it_matters too short")


class VerdictValidator:
    USE_MODES = {"support", "contrast", "harmful"}
    PARTIES = {"claimant", "respondent", "mixed", "unclear"}
    OUTCOMES = {"allowed", "dismissed", "remitted", "mixed", "unknown"}

    @staticmethod
    def validate(d: Dict[str, Any]) -> None:
        if not isinstance(d, dict):
            raise TypeError("Verdict must be a dict")

        for k in ("atom_id", "doc_id", "relevant"):
            if k not in d:
                raise ValueError(f"Verdict missing required field: {k}")

        if not str(d["atom_id"]).strip():
            raise ValueError("Verdict.atom_id must be non-empty")
        if not str(d["doc_id"]).strip():
            raise ValueError("Verdict.doc_id must be non-empty")

        # matched_X
        mx = d.get("matched_X", [])
        if mx is not None and not isinstance(mx, list):
            raise ValueError("Verdict.matched_X must be a list")
        for x in (mx or []):
            if not str(x).strip().startswith("X"):
                raise ValueError(f"Verdict.matched_X contains invalid: {x}")

        # ints 0-100
        for k in ("precedent_score", "confidence"):
            if k in d and d[k] is not None:
                try:
                    v = int(d[k])
                except Exception:
                    raise ValueError(f"Verdict.{k} must be int")
                if v < 0 or v > 100:
                    raise ValueError(f"Verdict.{k} must be 0..100")

        # anchors
        anchors = d.get("anchors", [])
        if anchors is not None and not isinstance(anchors, list):
            raise ValueError("Verdict.anchors must be a list")
        for a in (anchors or []):
            AnchorValidator.validate(a)

        um = d.get("use_mode", "contrast")
        if um not in VerdictValidator.USE_MODES:
            raise ValueError(f"Verdict.use_mode invalid: {um}")

        pw = d.get("proposition_winner", "unclear")
        if pw not in VerdictValidator.PARTIES:
            raise ValueError(f"Verdict.proposition_winner invalid: {pw}")

        ao = d.get("appeal_outcome", "unknown")
        if ao not in VerdictValidator.OUTCOMES:
            raise ValueError(f"Verdict.appeal_outcome invalid: {ao}")

        sp = d.get("successful_party", "unclear")
        if sp not in VerdictValidator.PARTIES:
            raise ValueError(f"Verdict.successful_party invalid: {sp}")

        # distinguishers
        dist = d.get("distinguishers", [])
        if dist is not None and not isinstance(dist, list):
            raise ValueError("Verdict.distinguishers must be a list")
