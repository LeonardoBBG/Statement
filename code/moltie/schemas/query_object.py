from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Y JSON not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))

def save_json(obj: Dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def build_deduped_y_view(
    y_json: Dict[str, Any],
    collapsed_evidence_to_x: Dict[str, str],
) -> Dict[str, Any]:
    """
    Return a Y JSON "view" that contains only the Xi that survived evidence-based dedup.
    Keeps original Xi bodies but filters to the kept set.
    """
    kept_x = sorted(set(collapsed_evidence_to_x.values()))
    x_tests_full = y_json.get("x_tests", {}) or {}

    deduped = dict(y_json)  # shallow copy
    deduped["x_tests"] = {x: x_tests_full[x] for x in kept_x if x in x_tests_full}

    # Optional: embed audit
    deduped["dedup_audit"] = {
        "rule": "one_evidence_one_X",
        "collapsed_evidence_to_x": collapsed_evidence_to_x,
        "kept_x_tests": kept_x,
    }
    return deduped

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


# ----------------------------
# Downstream semantic de-dup helpers (lean, deterministic)
# ----------------------------

def _safe_list(x: Any) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    return [str(x).strip()] if str(x).strip() else []


def x_specificity_score(y_json: Dict[str, Any], x_id: str) -> int:
    """
    Deterministic specificity score.
    Higher = more specific (wins conflicts).

    Priority:
      1) required_elements length (v2 schema)
      2) positive_indicators length
      3) excludes length
    """
    x_tests = (y_json or {}).get("x_tests") or {}
    t = x_tests.get(x_id) or {}
    req = _safe_list(t.get("required_elements"))
    pos = _safe_list(t.get("positive_indicators"))
    exc = _safe_list(t.get("excludes"))
    return (1000 * len(req)) + (10 * len(pos)) + len(exc)


def select_most_specific_x(y_json: Dict[str, Any], x_ids: List[str]) -> str:
    """
    Pick exactly one Xi from a conflicting set.
    Most specific wins. Ties broken by lexical order for determinism.
    """
    xs = [str(x).strip() for x in (x_ids or []) if str(x).strip()]
    if not xs:
        raise ValueError("select_most_specific_x: empty x_ids")

    # Deterministic: score desc, then X-id asc
    xs_sorted = sorted(xs, key=lambda x: (-x_specificity_score(y_json, x), x))
    return xs_sorted[0]


def collapse_x_hits_one_per_evidence(
    y_json: Dict[str, Any],
    evidence_to_x: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    Core rule:
      One evidence_id -> one Xi

    Input:
      evidence_to_x = { "evidence_id_1": ["X1","X3"], "evidence_id_2": ["X4"] }

    Output:
      { "evidence_id_1": "X3", "evidence_id_2": "X4" }
    """
    if not isinstance(evidence_to_x, dict):
        raise TypeError("evidence_to_x must be a dict[evidence_id -> list[x_id]]")

    collapsed: Dict[str, str] = {}
    for evidence_id, xs in evidence_to_x.items():
        x_list = [str(x).strip() for x in (xs or []) if str(x).strip()]
        if not x_list:
            continue
        collapsed[str(evidence_id)] = select_most_specific_x(y_json, x_list)
    return collapsed


def unique_x_tests_from_collapsed(collapsed: Dict[str, str]) -> List[str]:
    """
    Convert collapsed mapping (evidence_id -> Xi) into a stable unique list of Xi.
    Sorted by Xi ID for determinism.
    """
    xs = sorted({str(x).strip() for x in (collapsed or {}).values() if str(x).strip()})
    return xs


def merge_indicators_and_excludes(y_json: Dict[str, Any], x_tests: List[str]) -> Dict[str, List[str]]:
    """
    Merge positive_indicators and excludes across selected Xi (dedup, preserve order).
    """
    out_pos: List[str] = []
    out_exc: List[str] = []

    x_map = (y_json or {}).get("x_tests") or {}
    for x_id in (x_tests or []):
        t = x_map.get(x_id) or {}
        for p in _safe_list(t.get("positive_indicators")):
            if p not in out_pos:
                out_pos.append(p)
        for e in _safe_list(t.get("excludes")):
            if e not in out_exc:
                out_exc.append(e)

    return {"positive_indicators": out_pos, "excludes": out_exc}

def build_atom_from_y_path(
    *,
    atom_id: str,
    y_path: str,
    evidence_to_x: Dict[str, List[str]],
    dedup_out_path: Optional[str] = None,
) -> Tuple[AtomQuery, Dict[str, Any]]:
    """
    Loads Y from disk, applies evidence-based dedup, returns:
      (AtomQuery, deduped_Y_json)

    dedup_out_path: if provided, writes deduped Y to disk.
    """
    y_json = load_json(y_path)

    collapsed = collapse_x_hits_one_per_evidence(y_json, evidence_to_x)

    atom = build_atom_query_from_hits(
        atom_id=atom_id,
        y_json=y_json,
        evidence_to_x=evidence_to_x,
        proposition=None,  # auto-derive from Xi names
    )

    deduped_y = build_deduped_y_view(y_json, collapsed)

    if dedup_out_path:
        save_json(deduped_y, dedup_out_path)

    return atom, deduped_y


def build_atom_query_from_hits(
    *,
    atom_id: str,
    y_json: Dict[str, Any],
    evidence_to_x: Dict[str, List[str]],
    proposition: Optional[str] = None,
) -> AtomQuery:
    """
    Convenience builder:
    - applies one-evidence->one-X rule
    - derives x_tests list
    - merges indicators/excludes

    You can call this from your objective function builder after you generate evidence hits.
    """
    collapsed = collapse_x_hits_one_per_evidence(y_json, evidence_to_x)
    x_tests = unique_x_tests_from_collapsed(collapsed)

    if not proposition:
        # Compact one-liner from selected Xi names
        x_map = (y_json or {}).get("x_tests") or {}
        names = [str((x_map.get(x) or {}).get("name", x)).strip() for x in x_tests]
        names = [n for n in names if n]
        proposition = " / ".join(names) if names else f"{atom_id}: inferred proposition"

    merged = merge_indicators_and_excludes(y_json, x_tests)

    return AtomQuery(
        atom_id=str(atom_id).strip(),
        x_tests=x_tests,
        proposition=str(proposition).strip(),
        positive_indicators=merged["positive_indicators"],
        excludes=merged["excludes"],
        keyword_seeds=[],
        expansion_terms=[],
    )
