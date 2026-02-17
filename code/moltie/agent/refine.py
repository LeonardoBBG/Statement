from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional, List

from ..schemas.query_object import AtomQuery
from ..schemas.verdict import Verdict

@dataclass(frozen=True)
class RefineOutput:
    atom: AtomQuery
    changed: bool
    note: str

# Conservative expansions that are "meaning-preserving" for common signals.
# Keep this small; add atom-specific maps later.
_GENERIC_EXPANSIONS = {
    "predetermined": ["foregone conclusion", "already decided", "decision already taken", "no real appeal"],
    "appeal": ["appeal process", "appeal hearing", "appeal outcome", "internal appeal"],
    "final": ["treated as final", "fait accompli", "final decision", "decision was final"],
    "allegation": ["accusation", "complaint", "claim made", "reported allegation"],
    "as fact": ["presented as fact", "stated as fact", "assumed true", "treated as true"],
    "investigation": ["inquiry", "fact-finding", "investigatory", "looked into"],
    "undermined": ["compromised", "tainted", "not genuine", "not impartial"],
}

def _norm_terms(xs: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in xs or []:
        t = (x or "").strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out

def _add_terms(base: List[str], add: List[str], cap: int) -> List[str]:
    base_n = _norm_terms(base)
    add_n = _norm_terms(add)
    merged = base_n + [t for t in add_n if t.lower() not in {b.lower() for b in base_n}]
    return merged[:cap]

def refine_query(atom: AtomQuery, verdict: Verdict, iter_no: int) -> RefineOutput:
    """
    v1: deterministic recall-surface refinement.
    - If no anchors / irrelevant / low confidence: expand keyword_seeds & expansion_terms
    - Escalate gently by iteration number.
    """
    # If we've already refined a few times, stop bloating the query.
    if iter_no >= 4:
        return RefineOutput(atom=atom, changed=False, note="Refine capped (iter>=4)")

    # Decide if we should refine based on verifier outcome
    # (adjust fields if your Verdict schema differs)
    relevant = bool(getattr(verdict, "relevant", False))
    anchors = getattr(verdict, "anchors", None) or []
    conf = getattr(verdict, "confidence", 0) or 0

    should_refine = (not relevant) or (not anchors) or (conf < 60)
    if not should_refine:
        return RefineOutput(atom=atom, changed=False, note="No refine needed (good signal)")

    # Build candidate expansions from the atom's own indicators and proposition
    seeds_src = []
    seeds_src += [atom.proposition] if getattr(atom, "proposition", None) else []
    seeds_src += list(getattr(atom, "positive_indicators", []) or [])
    seeds_src += list(getattr(atom, "keyword_seeds", []) or [])

    # Simple: pick a few anchor-ish keywords from those strings (no regex heroics).
    # We keep it conservative: only match expansions when a key substring appears.
    extra = []
    joined = " | ".join([s.lower() for s in seeds_src if isinstance(s, str)])
    for k, v in _GENERIC_EXPANSIONS.items():
        if k in joined:
            extra.extend(v)

    # Escalation schedule
    # iter 0: add a couple generic expansions
    # iter 1: add more expansions
    # iter 2+: also add "negative cues" into excludes only if atom already has excludes list
    cap_keywords = 10 + iter_no * 5
    cap_expansions = 12 + iter_no * 6

    new_keyword_seeds = _add_terms(getattr(atom, "keyword_seeds", []) or [], extra[: 3 + iter_no * 2], cap_keywords)
    new_expansion_terms = _add_terms(getattr(atom, "expansion_terms", []) or [], extra, cap_expansions)

    changed = (new_keyword_seeds != (getattr(atom, "keyword_seeds", []) or [])) or (
        new_expansion_terms != (getattr(atom, "expansion_terms", []) or [])
    )

    if not changed:
        return RefineOutput(atom=atom, changed=False, note="Refine attempted; no new terms available")

    new_atom = replace(
        atom,
        keyword_seeds=new_keyword_seeds,
        expansion_terms=new_expansion_terms,
    )
    return RefineOutput(atom=new_atom, changed=True, note=f"Refined recall terms (iter={iter_no})")
