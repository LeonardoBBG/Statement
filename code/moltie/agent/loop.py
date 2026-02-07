import hashlib
import json
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
    trace: List[Dict[str, Any]]


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:10]


def run_agent_on_one_doc(
    doc_id: str,
    paras: List[Dict[str, str]],
    atom: AtomQuery,
    run_cfg: RunConfig,
    llm_cfg: LLMClientConfig,
) -> LoopResult:

    # ---- FAIL-FAST invariants (professional debugging) ----
    assert isinstance(doc_id, str) and doc_id.strip(), "doc_id must be non-empty"
    assert isinstance(paras, list) and len(paras) > 0, "paras must be a non-empty list"
    assert getattr(atom, "atom_id", "").strip(), "AtomQuery.atom_id is empty (fix AtomQuery construction)"

    trace: List[Dict[str, Any]] = []
    best: Optional[Verdict] = None
    best_score = -1
    plateau = 0

    # single compact entry log
    print(f"[moltie.loop] start doc_id={doc_id!r} atom_id={atom.atom_id!r} n_paras={len(paras)}")

    for i in range(1, run_cfg.max_iters + 1):
        rr = retrieve_evidence(doc_id=doc_id, paras=paras, atom=atom, k=run_cfg.k_chunks_per_doc)

        n_rr = len(rr.paras or [])
        if n_rr == 0:
            print(f"[moltie.loop] iter={i} WARN retrieval returned 0 paras meta={rr.retrieval!r}")

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
        # PATCH 1: deterministic anchor verification (verbatim + para_id)
        # =========================
        para_map = {p["para_id"]: p["text"] for p in evidence_pack["paras"]}

        bad = []
        for a in verdict.anchors:
            pid = a.para_id if hasattr(a, "para_id") else a.get("para_id")
            q   = a.quote   if hasattr(a, "quote")   else a.get("quote")

            if not pid or pid not in para_map:
                bad.append(("bad_para_id", pid))
                continue
            if not q or q not in para_map[pid]:
                bad.append(("non_verbatim", pid))

        if bad:
            # Force retry/repair path by raising (will be caught by outer try only if you wrap this)
            # Since we're *after* verify_with_ollama, treat as hard failure.
            print(f"[moltie.loop] iter={i} BAD_ANCHORS {bad[:3]} (showing up to 3)")
            raise ValueError(f"Invalid anchors (non-verbatim or bad para_id): {bad[:3]}")

        trace.append({"iter": i, "retrieval": rr.retrieval, "verdict": verdict.to_dict()})

        if verdict.precedent_score > best_score:
            best, best_score = verdict, verdict.precedent_score
            plateau = 0
        else:
            plateau += 1

        # =========================
        # PATCH 2: confidence scale normalization (only if thresh_conf is 0..1)
        # =========================
        conf = verdict.confidence
        if 0 <= run_cfg.thresh_conf <= 1:
            conf = verdict.confidence / 100.0

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

        if plateau >= run_cfg.plateau_p:
            print(f"[moltie.loop] EXIT plateau iter={i} plateau={plateau} best_score={best_score}")
            nx = NegativeExit.from_best(
                atom_id=atom.atom_id,
                reason="plateau",
                best=best,
                note=f"Plateau for {plateau} iterations; best_score={best_score}",
            )
            return LoopResult(verdict=None, negative_exit=nx, iters=i, trace=trace)

        atom = refine_query(atom, verdict, i).atom

    print(f"[moltie.loop] EXIT exhausted iters={run_cfg.max_iters} best_score={best_score}")
    nx = NegativeExit.from_best(
        atom_id=atom.atom_id,
        reason="exhausted",
        best=best,
        note=f"Max iters reached; best_score={best_score}",
    )
    return LoopResult(verdict=None, negative_exit=nx, iters=run_cfg.max_iters, trace=trace)
