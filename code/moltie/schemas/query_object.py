from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ----------------------------
# Query object: runtime intent derived from Y_JSON
# ----------------------------

@dataclass(frozen=True)
class AtomQuery:
    """
    One runtime query unit derived from Y_JSON.

    Design goal:
    - compact (no WS prose)
    - expandable (keywords can grow across iterations)
    - stable IDs for audit
    """
    atom_id: str
    x_tests: List[str]                          # e.g. ["X1","X2","X3","X4"]
    proposition: str                            # one-liner, derived from X names/definitions
    positive_indicators: List[str]              # merged indicators across selected X tests
    excludes: List[str] = field(default_factory=list)

    # runtime / iterative fields
    keyword_seeds: List[str] = field(default_factory=list)
    expansion_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AtomQuery":
        AtomQueryValidator.validate(d)
        return AtomQuery(
            atom_id=str(d["atom_id"]).strip(),
            x_tests=list(d["x_tests"]),
            proposition=str(d["proposition"]).strip(),
            positive_indicators=list(d["positive_indicators"]),
            excludes=list(d.get("excludes", [])),
            keyword_seeds=list(d.get("keyword_seeds", [])),
            expansion_terms=list(d.get("expansion_terms", [])),
        )


@dataclass(frozen=True)
class QueryObject:
    """
    Container for all atom queries for one run.
    """
    version: str
    atoms: List[AtomQuery]
    source_y_version: Optional[str] = None      # keep link to Y_JSON version if you have it

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "source_y_version": self.source_y_version,
            "atoms": [a.to_dict() for a in self.atoms],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "QueryObject":
        QueryObjectValidator.validate(d)
        atoms = [AtomQuery.from_dict(x) for x in d["atoms"]]
        return QueryObject(
            version=str(d["version"]).strip(),
            source_y_version=(str(d["source_y_version"]).strip() if d.get("source_y_version") else None),
            atoms=atoms,
        )


# ----------------------------
# Validators (strict, dependency-free)
# ----------------------------

class AtomQueryValidator:
    @staticmethod
    def validate(d: Dict[str, Any]) -> None:
        if not isinstance(d, dict):
            raise TypeError("AtomQuery must be a dict")
        req = ["atom_id", "x_tests", "proposition", "positive_indicators"]
        for k in req:
            if k not in d:
                raise ValueError(f"AtomQuery missing required field: {k}")

        atom_id = str(d["atom_id"]).strip()
        if not atom_id:
            raise ValueError("AtomQuery.atom_id must be non-empty")

        x_tests = d["x_tests"]
        if not isinstance(x_tests, list) or not x_tests:
            raise ValueError("AtomQuery.x_tests must be a non-empty list")
        for x in x_tests:
            xs = str(x).strip()
            if not xs.startswith("X"):
                raise ValueError(f"AtomQuery.x_tests contains invalid value: {x}")

        prop = str(d["proposition"]).strip()
        if len(prop) < 10:
            raise ValueError("AtomQuery.proposition too short (needs meaningful one-liner)")

        pos = d["positive_indicators"]
        if not isinstance(pos, list) or not pos:
            raise ValueError("AtomQuery.positive_indicators must be a non-empty list")
        for p in pos:
            if len(str(p).strip()) < 3:
                raise ValueError("AtomQuery.positive_indicators contains empty/too-short item")

        for opt_list_key in ("excludes", "keyword_seeds", "expansion_terms"):
            if opt_list_key in d and not isinstance(d[opt_list_key], list):
                raise ValueError(f"AtomQuery.{opt_list_key} must be a list if present")


class QueryObjectValidator:
    @staticmethod
    def validate(d: Dict[str, Any]) -> None:
        if not isinstance(d, dict):
            raise TypeError("QueryObject must be a dict")
        if "version" not in d:
            raise ValueError("QueryObject missing required field: version")
        if "atoms" not in d:
            raise ValueError("QueryObject missing required field: atoms")

        v = str(d["version"]).strip()
        if not v:
            raise ValueError("QueryObject.version must be non-empty")

        atoms = d["atoms"]
        if not isinstance(atoms, list) or not atoms:
            raise ValueError("QueryObject.atoms must be a non-empty list")

        # validate each atom
        for a in atoms:
            AtomQueryValidator.validate(a)
