from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from ..schemas.query_object import AtomQuery
from ..schemas.verdict import Verdict

@dataclass(frozen=True)
class RefineOutput:
    atom: AtomQuery
    changed: bool
    note: str

def refine_query(atom: AtomQuery, verdict: Verdict, iter_no: int) -> RefineOutput:
    """
    MVP: no refinement yet.
    Later: expand keyword_seeds / expansion_terms based on verifier feedback.
    """
    return RefineOutput(atom=atom, changed=False, note="MVP: no refine")
