from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..schemas.query_object import AtomQuery
from ..schemas.run_config import RunConfig
from ..schemas.verdict import Verdict
from ..schemas.negative_exit import NegativeExit
from ..llm.verifier_prompt import build_verifier_prompt
from ..llm.client import verify_with_ollama, LLMClientConfig

from .retrieve import retrieve_evidence
from .refine import refine_query


@dataclass
class LoopResult:
    verdict: Optional[Verdict]
    negative_exit: Optional[NegativeExit]
    iters: int
    trace: List[Dict[str, Any]]          # audit of each iteration


def run_agent_on_one_doc(
    doc_id: str,
    paras: List[Dict[str, str]],
    atom: AtomQuery,
    run_cfg: RunConfig,
    llm_cfg: LLMClientConfig,
) -> LoopResult:
    """
    Minimal agent loop:
    - retrieve evidence pack
    - verify
    - if meets thresholds => accept
    - else refine (stub) or negative exit on plateau/exhausted
    """
    trace: List[Dict[str, Any]] = []
    best: Optional[Verdict] = None
    best_score = -1

    plateau = 0

    for i in range(1, run_cfg.max_iters + 1):
        rr = retrieve_evidence(
            doc_id=doc_id,
            paras=paras,
            atom=atom,
            k=run_cfg.k_chunks_per_doc,
        )

        evidence_pack = {
            "doc_id": rr.doc_id,
            "doc_meta": {},
            "paras": rr.paras,
            "retrieval": rr.retrieval,
        }

        prompt = build_verifier_prompt(atom=atom, evidence_pack=evidence_pack, cfg=run_cfg)
        verdict = verify_with_ollama(prompt, llm_cfg)

        trace.append({
            "iter": i,
            "retrieval": rr.retrieval,
            "verdict": verdict.to_dict(),
        })

        # track best
        if verdict.precedent_score > best_score:
            best, best_score = verdict, verdict.precedent_score
            plateau = 0
        else:
            plateau += 1

        # accept condition (objective function)
        if (verdict.relevant
            and verdict.precedent_score >= run_cfg.thresh_score
            and verdict.confidence >= run_cfg.thresh_conf
            and len(verdict.anchors) >= run_cfg.anchors_required):
            return LoopResult(verdict=verdict, negative_exit=None, iters=i, trace=trace)

        # negative outlet: plateau
        if plateau >= run_cfg.plateau_p:
            nx = NegativeExit.from_best(
                atom_id=atom.atom_id,
                reason="plateau",
                best=best,
                note=f"Plateau for {plateau} iterations; best_score={best_score}",
            )
            return LoopResult(verdict=None, negative_exit=nx, iters=i, trace=trace)

        # refine (currently no-op)
        ro = refine_query(atom, verdict, i)
        atom = ro.atom

    nx = NegativeExit.from_best(
        atom_id=atom.atom_id,
        reason="exhausted",
        best=best,
        note=f"Max iters reached; best_score={best_score}",
    )
    return LoopResult(verdict=None, negative_exit=nx, iters=run_cfg.max_iters, trace=trace)
