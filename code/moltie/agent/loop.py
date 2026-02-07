from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from ..schemas.query_object import AtomQuery
from ..schemas.run_config import RunConfig
from ..schemas.verdict import Verdict
from ..schemas.negative_exit import NegativeExit
from ..llm.verifier_prompt import build_verifier_prompt
from ..llm.client import verify_with_ollama, LLMClientConfig

from .retrieve import retrieve_windowed_evidence  # <-- you must add this in retrieve.py
from .refine import refine_query


@dataclass
class LoopResult:
    verdict: Optional[Verdict]
    negative_exit: Optional[NegativeExit]
    iters: int
    trace: List[Dict[str, Any]]


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:10]


def _compute_quality(v: Verdict) -> int:
    """
    Plateau metric (NOT acceptance metric).
    Must be monotonic with "signal getting better".
    """
    rel = 1 if getattr(v, "relevant", False) else 0
    anchors_n = len(getattr(v, "anchors", []) or [])
    ps = int(getattr(v, "precedent_score", 0) or 0)
    conf = int(getattr(v, "confidence", 0) or 0)
    # Weight precedent_score most, but reward relevance, anchors, confidence.
    return (ps * 10) + (rel * 30) + (anchors_n * 8) + (conf // 10)


def run_agent_on_one_doc(
    doc_id: str,
    paras: List[Dict[str, str]],
    atom: AtomQuery,
    run_cfg: RunConfig,
    llm_cfg: LLMClientConfig,
) -> LoopResult:

    # ---- FAIL-FAST invariants ----
    assert isinstance(doc_id, str) and doc_id.strip(), "doc_id must be non-empty"
    assert isinstance(paras, list) and len(paras) > 0, "paras must be a non-empty list"
    assert getattr(atom, "atom_id", "").strip(), "AtomQuery.atom_id is empty (fix AtomQuery construction)"

    trace: List[Dict[str, Any]] = []
    best: Optional[Verdict] = None
    best_quality = -1
    best_score = -1
    plateau = 0

    # Coverage state: prevents rereading the same region forever
    visited_starts: Set[int] = set()

    print(f"[moltie.loop] start doc_id={doc_id!r} atom_id={atom.atom_id!r} n_paras={len(paras)}")

    # Window params (use RunConfig if you add them; otherwise defaults)
    window_size = getattr(run_cfg, "window_size", 24)
    stride = getattr(run_cfg, "stride", 12)
    top_windows = getattr(run_cfg, "top_windows", 3)
    min_hits = getattr(run_cfg, "min_hits", 2)

    # How many distinct window starts exist?
    est_total_windows = max(1, (len(paras) + stride - 1) // max(1, stride))

    for i in range(1, run_cfg.max_iters + 1):
        rr = retrieve_windowed_evidence(
            doc_id=doc_id,
            paras=paras,
            atom=atom,
            k=run_cfg.k_chunks_per_doc,
            min_hits=min_hits,
            window_size=window_size,
            stride=stride,
            top_windows=top_windows,
            visited_starts=visited_starts,
            ensure_coverage=True,
        )

        n_rr = len(rr.paras or [])
        if n_rr == 0:
            print(f"[moltie.loop] iter={i} WARN retrieval returned 0 paras meta={rr.retrieval!r}")

        # Mark visited windows (so next iteration moves)
        picked = rr.retrieval.get("picked_windows") or []
        for w in picked:
            s = w.get("start")
            if isinstance(s, int):
                visited_starts.add(s)

        evidence_pack = {"doc_id": rr.doc_id, "doc_meta": {}, "paras": rr.paras, "retrieval": rr.retrieval}
        prompt = build_verifier_prompt(atom=atom, evidence_pack=evidence_pack, cfg=run_cfg)

        if atom.atom_id not in prompt:
            print(f"[moltie.loop] iter={i} WARN atom_id not found in prompt prompt_hash={_h(prompt)}")

        try:
            verdict = verify_with_ollama(prompt, llm_cfg)
        except Exception as e:
            print(
                "[moltie.loop] LLM_FAIL "
                f"iter={i} doc_id={doc_id!r} atom_id={atom.atom_id!r} "
                f"n_rr_paras={n_rr} prompt_hash={_h(prompt)} err={type(e).__name__}: {e}"
            )
            raise

        if not getattr(verdict, "atom_id", "").strip():
            print(f"[moltie.loop] iter={i} BAD_VERDICT missing atom_id prompt_hash={_h(prompt)} verdict={verdict.to_dict()}")
            raise RuntimeError("LLM returned Verdict without atom_id")

        # =========================
        # Anchor verification: verbatim + para_id
        # =========================
        para_map = {p["para_id"]: p["text"] for p in evidence_pack["paras"]}

        bad = []
        for a in verdict.anchors:
            pid = a.para_id if hasattr(a, "para_id") else a.get("para_id")
            q = a.quote if hasattr(a, "quote") else a.get("quote")

            if not pid or pid not in para_map:
                bad.append(("bad_para_id", pid))
                continue
            if not q or q not in para_map[pid]:
                bad.append(("non_verbatim", pid))

        if bad:
            print(f"[moltie.loop] iter={i} BAD_ANCHORS {bad[:3]} (showing up to 3)")
            raise ValueError(f"Invalid anchors (non-verbatim or bad para_id): {bad[:3]}")

        trace.append({"iter": i, "retrieval": rr.retrieval, "verdict": verdict.to_dict()})

        # =========================
        # Plateau tracking uses eps_improve (your RunConfig)
        # =========================
        q = _compute_quality(verdict)
        if q >= (best_quality + int(run_cfg.eps_improve)):
            best, best_quality = verdict, q
            best_score = max(best_score, int(getattr(verdict, "precedent_score", 0) or 0))
            plateau = 0
        else:
            plateau += 1

        # =========================
        # confidence normalization (keep your current semantics)
        # =========================
        conf = verdict.confidence
        if 0 <= run_cfg.thresh_conf <= 1:
            conf = verdict.confidence / 100.0

        # =========================
        # ACCEPT (unchanged)
        # =========================
        if (
            verdict.relevant
            and verdict.precedent_score >= run_cfg.thresh_score
            and conf >= run_cfg.thresh_conf
            and len(verdict.anchors) >= run_cfg.anchors_required
        ):
            print(
                f"[moltie.loop] ACCEPT iter={i} score={verdict.precedent_score} "
                f"conf={verdict.confidence} anchors={len(verdict.anchors)}"
            )
            return LoopResult(verdict=verdict, negative_exit=None, iters=i, trace=trace)

        # =========================
        # PLATEAU: MOVE before EXIT (coverage first)
        # =========================
        if plateau >= run_cfg.plateau_p:
            if len(visited_starts) < est_total_windows:
                print(
                    f"[moltie.loop] PLATEAU iter={i} but continuing for coverage "
                    f"(plateau={plateau} visited_windows={len(visited_starts)}/{est_total_windows})"
                )
                plateau = 0
                atom = refine_query(atom, verdict, i).atom
                continue

            print(
                f"[moltie.loop] EXIT plateau iter={i} plateau={plateau} "
                f"best_quality={best_quality} best_score={best_score} visited_windows={len(visited_starts)}"
            )
            nx = NegativeExit.from_best(
                atom_id=atom.atom_id,
                reason="plateau",
                best=best,
                note=(
                    f"Plateau after coverage; visited_windows={len(visited_starts)}/"
                    f"{est_total_windows} best_quality={best_quality} best_score={best_score}"
                ),
            )
            return LoopResult(verdict=None, negative_exit=nx, iters=i, trace=trace)

        atom = refine_query(atom, verdict, i).atom

    print(f"[moltie.loop] EXIT exhausted iters={run_cfg.max_iters} best_quality={best_quality} best_score={best_score}")
    nx = NegativeExit.from_best(
        atom_id=atom.atom_id,
        reason="exhausted",
        best=best,
        note=f"Max iters reached; best_quality={best_quality} best_score={best_score}",
    )
    return LoopResult(verdict=None, negative_exit=nx, iters=run_cfg.max_iters, trace=trace)
