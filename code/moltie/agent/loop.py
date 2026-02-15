from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from ..llm.client import LLMClientConfig, verify_with_ollama
from ..llm.verifier_prompt import build_verifier_prompt
from ..schemas.negative_exit import NegativeExit
from ..schemas.query_object import AtomQuery
from ..schemas.run_config import RunConfig
from ..schemas.verdict import Verdict

from .refine import refine_query
from .retrieve import retrieve_windowed_evidence


@dataclass
class LoopResult:
    verdict: Optional[Verdict]
    negative_exit: Optional[NegativeExit]
    iters: int
    trace: List[Dict[str, Any]]


_WS = re.compile(r"\s+")
_ZW = re.compile(r"[\u200b\u200c\u200d\ufeff]")  # zero-width chars
_DASH = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2212]")  # hyphen/dash variants

_HARD_JUNK_SIGNALS = (
    "neutral citation",
    "royal courts of justice",
    "in the supreme court of judicature",
    "court of appeal",
    "civil division",
    "on appeal from",
    "employment appeal tribunal",
)

_SOFT_JUNK_SIGNALS = (
    "before:",
    "between:",
    "hearing date",
    "transcript",
)

def _is_junk_para(text: str) -> bool:
    if not text:
        return True

    s_raw = text
    s = _norm(text).lower()

    # compute density FIRST (used by hard + soft kill)
    letters = sum(ch.isalpha() for ch in s_raw)
    letter_ratio = letters / max(1, len(s_raw))

    # --- Hard kill: only if header-like ---
    if any(sig in s for sig in _HARD_JUNK_SIGNALS):
        if len(s_raw) < 400 or letter_ratio < 0.55:
            return True

    # --- Soft kill: docket-format paragraphs (low semantic density) ---
    if len(s_raw) >= 160 and letter_ratio < 0.42 and len(s_raw) < 3000:
        return True

    # "Before/Between" blocks are usually parties/judges; kill if not huge
    if any(sig in s for sig in _SOFT_JUNK_SIGNALS) and len(s_raw) < 2200:
        return True

    return False

def _filter_front_matter(paras, min_keep=4, debug=False, head_limit=40):
    if not paras:
        return paras

    before = len(paras)

    head = paras[:head_limit]
    tail = paras[head_limit:]  # NEVER filter the tail

    head_f = [p for p in head if not _is_junk_para((p.get("text") or ""))]

    filtered = head_f + tail

    # guardrail: don’t let it nuke the doc
    if len(filtered) < min_keep:
        filtered = paras

    if debug:
        print(
            f"[moltie.debug] front_matter_filter: before={before} after={len(filtered)} "
            f"dropped={before-len(filtered)} head_limit={head_limit}"
        )
    return filtered

def _norm(s: str) -> str:
    if not s:
        return ""
    s = _ZW.sub("", s)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = _DASH.sub("-", s)
    s = s.replace("\u00a0", " ")  # nbsp
    s = _WS.sub(" ", s).strip()
    return s


def _quote_ok(quote: str, text: str) -> bool:
    if not quote or not text:
        return False

    # 1) strict
    if quote in text:
        return True

    # 2) normalized strict (punctuation/whitespace/quotes/dashes)
    nq = _norm(quote)
    nt = _norm(text)
    if nq and nq in nt:
        return True

    return False

def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:10]


def _nn_int(x: Any) -> int:
    """Non-negative int coercion."""
    try:
        v = int(x)
        return v if v > 0 else 0
    except Exception:
        return 0


def _compute_quality(v: Verdict) -> int:
    """Plateau metric (NOT acceptance metric)."""
    rel = 1 if getattr(v, "relevant", False) else 0
    anchors_n = len(getattr(v, "anchors", []) or [])
    ps = _nn_int(getattr(v, "precedent_score", 0))
    conf = _nn_int(getattr(v, "confidence", 0))
    return (ps * 10) + (rel * 30) + (anchors_n * 8) + (conf // 10)


def _anchors_valid(verdict: Verdict, para_map: Dict[str, str]) -> Tuple[bool, List[Tuple[str, Optional[str]]]]:
    bad: List[Tuple[str, Optional[str]]] = []

    anchors = getattr(verdict, "anchors", []) or []
    for a in anchors:
        pid = a.para_id if hasattr(a, "para_id") else a.get("para_id")
        q = a.quote if hasattr(a, "quote") else a.get("quote")

        if not pid or pid not in para_map:
            bad.append(("bad_para_id", pid))
            continue
        if not q or not _quote_ok(q, para_map[pid]):
            bad.append(("non_verbatim", pid))

    return (len(bad) == 0, bad)

def _evidence_to_x_from_verdict(doc_id: str, verdict: Verdict) -> Dict[str, List[str]]:
    """
    Build evidence_to_x at PARA granularity:
      evidence_id = f"{doc_id}:{para_id}"
      value = list of X ids (from verdict.matched_X)
    """
    xs = [str(x).strip() for x in (getattr(verdict, "matched_X", None) or []) if str(x).strip()]
    anchors = getattr(verdict, "anchors", None) or []

    out: Dict[str, Set[str]] = {}
    for a in anchors:
        pid = a.para_id if hasattr(a, "para_id") else a.get("para_id")
        if not pid:
            continue
        evid = f"{doc_id}:{str(pid).strip()}"
        out.setdefault(evid, set()).update(xs)

    return {k: sorted(v) for k, v in out.items()}


def _is_strong_for_harvest(v: Verdict, run_cfg: RunConfig) -> bool:
    """Strong signal filter for harvest memory (separate from ACCEPT thresholds)."""
    min_score = _nn_int(getattr(run_cfg, "harvest_min_score", 60))
    min_conf = _nn_int(getattr(run_cfg, "harvest_min_conf", 70))
    min_anchors = _nn_int(getattr(run_cfg, "harvest_min_anchors", 1))

    return (
        bool(getattr(v, "relevant", False))
        and _nn_int(getattr(v, "precedent_score", 0)) >= min_score
        and _nn_int(getattr(v, "confidence", 0)) >= min_conf
        and len(getattr(v, "anchors", []) or []) >= min_anchors
    )


def _maybe_rechunk_single_blob_paras(
    paras: List[Dict[str, str]],
    *,
    min_len: int = 40,
    max_para_chars: int = 1800,
    blob_threshold: int = 5000,
) -> List[Dict[str, str]]:
    """
    E2E safety guard:
    If upstream PDF extraction collapsed everything into ONE giant paragraph,
    split it into multiple paras so anchors can reference valid para_ids.

    This does NOT depend on pdfplumber; it just restructures already-extracted text.
    """
    if not paras or len(paras) != 1:
        return paras

    text = (paras[0].get("text") or "").strip()
    if len(text) < blob_threshold:
        return paras  # not a huge blob; leave it

    # Normalize line endings
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Try paragraph split on blank lines first
    parts = [p.strip() for p in t.split("\n\n") if p.strip()]

    # If blank lines are missing, fall back to line grouping
    if len(parts) <= 3:
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        parts = []
        buf: List[str] = []
        size = 0
        for ln in lines:
            if size + len(ln) + 1 > max_para_chars and buf:
                parts.append(" ".join(buf).strip())
                buf = [ln]
                size = len(ln)
            else:
                buf.append(ln)
                size += len(ln) + 1
        if buf:
            parts.append(" ".join(buf).strip())

    # Filter tiny junk + cap long paras
    out: List[str] = []
    for p in parts:
        if len(p) < min_len:
            continue
        if len(p) <= max_para_chars:
            out.append(p)
        else:
            # Chunk long paras conservatively by sentences-ish
            chunk: List[str] = []
            cur = 0
            for sent in p.split(". "):
                s = sent if sent.endswith(".") else sent + "."
                if cur + len(s) + 1 > max_para_chars and chunk:
                    out.append(" ".join(chunk).strip())
                    chunk = [s]
                    cur = len(s)
                else:
                    chunk.append(s)
                    cur += len(s) + 1
            if chunk:
                out.append(" ".join(chunk).strip())

    return [{"para_id": f"p{i:05d}", "text": p} for i, p in enumerate(out, start=1)]


def run_harvest_then_reason_on_one_doc(
    doc_id: str,
    paras: List[Dict[str, str]],
    atom: "AtomQuery",
    run_cfg: "RunConfig",
    llm_cfg: "LLMClientConfig",
) -> "LoopResult":

    assert isinstance(doc_id, str) and doc_id.strip(), "doc_id must be non-empty"
    assert isinstance(paras, list) and len(paras) > 0, "paras must be a non-empty list"
    assert getattr(atom, "atom_id", "").strip(), "AtomQuery.atom_id is empty"

    # -----------------------------
    # SINGLE SOURCE OF TRUTH
    # -----------------------------
    debug = run_cfg.debug
    # -----------------------------

    paras = _maybe_rechunk_single_blob_paras(paras)

    print(f"[moltie.debug] raw_paras_before_filter={len(paras)}")
    if paras:
        print(f"[moltie.debug] raw_para0_len={len((paras[0].get('text') or ''))}")


    # Use the SAME debug variable
    paras = _filter_front_matter(paras, min_keep=4, debug=debug)

    print(f"[moltie.debug] paras_after_front_matter_filter={len(paras)}")


    trace: List[Dict[str, Any]] = []

    if debug:
        print(f"[moltie.loop] HARVEST start doc_id={doc_id!r} atom_id={atom.atom_id!r} n_paras={len(paras)}")

    if debug:
        print("[moltie.debug] raw paras:", len(paras), "sample lens:", [len((p.get("text") or "")) for p in paras[:3]])


    window_size = max(1, int(run_cfg.window_size))
    stride = max(1, int(run_cfg.stride))
    memory_max_items = int(run_cfg.memory_max_items)

    full_para_ids_in_order = [p["para_id"] for p in paras]
    full_para_map = {p["para_id"]: p["text"] for p in paras}

    harvested_verdicts: List["Verdict"] = []
    harvested_para_ids: Set[str] = set()
    harvested_windows: List[Dict[str, Any]] = []

    # --- ADDED: evidence_to_x accumulator (para-level evidence keys) ---
    # evidence_id format: f"{doc_id}:{para_id}"
    harvested_evidence_to_x: Dict[str, Set[str]] = {}

    # --- ADDED: local helper to bind anchors -> matched_X (no external deps) ---
    def _evidence_to_x_from_verdict(_doc_id: str, _verdict: "Verdict") -> Dict[str, List[str]]:
        xs = [str(x).strip() for x in (getattr(_verdict, "matched_X", None) or []) if str(x).strip()]
        anchors = getattr(_verdict, "anchors", None) or []
        out: Dict[str, Set[str]] = {}
        for a in anchors:
            pid = a.para_id if hasattr(a, "para_id") else a.get("para_id")
            if not pid:
                continue
            evid = f"{_doc_id}:{str(pid).strip()}"
            out.setdefault(evid, set()).update(xs)
        return {k: sorted(v) for k, v in out.items()}

    n = len(paras)

    if n <= window_size:
        starts = [0]
    else:
        starts = list(range(0, n, stride))
        last_start = max(0, n - window_size)
        if starts[-1] != last_start:
            starts.append(last_start)
        starts = sorted(set(starts))

    total_windows = len(starts)

    if debug:
        total_chars = sum(len(p.get("text") or "") for p in paras)
        print(
            f"[moltie.debug] doc totals: n_paras={n} total_chars={total_chars} "
            f"window_size={window_size} stride={stride} windows={total_windows}"
        )

    for start in starts:
        end = min(n, start + window_size)
        if end <= start:
            continue

        pack = paras[start:end]

        evidence_pack = {
            "doc_id": doc_id,
            "doc_meta": {},
            "paras": pack,
            "retrieval": {
                "method": "sweep",
                "start": start,
                "end": end,
                "window_size": window_size,
                "stride": stride,
            },
        }

        prompt = build_verifier_prompt(atom=atom, evidence_pack=evidence_pack, cfg=run_cfg)

        def _prev(s: str, n: int = 180) -> str:
            s = (s or "").replace("\n", " ").strip()
            return s[:n] + ("..." if len(s) > n else "")

        if run_cfg.debug:
            sent = evidence_pack["paras"] or []
            print("\n[moltie.forensic] ===== ABOUT TO CALL LLM =====")
            print("[moltie.forensic] iter:", i, "doc_id:", doc_id, "atom_id:", atom.atom_id, "x_tests:", getattr(atom, "x_tests", None))
            print("[moltie.forensic] total_paras_in_doc:", len(paras), "paras_sent:", len(sent))
            print("[moltie.forensic] retrieval_method:", (rr.retrieval or {}).get("method"), "picked_windows:", (rr.retrieval or {}).get("picked_windows"))

            # para_ids we are sending (first 12)
            ids = [p.get("para_id") for p in sent[:12]]
            print("[moltie.forensic] sent_para_ids_head:", ids)

            # show preview + lengths so we can see if we are sending mega blobs
            for p in sent[:3]:
                pid = p.get("para_id")
                txt = p.get("text") or ""
                print(f"[moltie.forensic] para {pid} len={len(txt)} preview={_prev(txt, 240)}")

            # if build_verifier_prompt embeds allowed ids, surface them
            # (we don't parse JSON fully; just detect whether the allowed ids appear at all)
            print("[moltie.forensic] prompt_chars:", len(prompt))
            print("[moltie.forensic] prompt_head_350:\n", prompt[:350])

            # quick sanity: do the para_ids we sent appear in the prompt at all?
            # (if not, the model is NOT seeing the para list you think it is)
            missing_in_prompt = [pid for pid in ids if pid and (pid not in prompt)]
            print("[moltie.forensic] sent_para_ids_missing_in_prompt_head:", missing_in_prompt[:8])

            print("[moltie.forensic] ===============================\n")


        verdict = verify_with_ollama(prompt, llm_cfg)

        pack_para_map = {p["para_id"]: p["text"] for p in pack}
        ok, bad = _anchors_valid(verdict, pack_para_map)

        strong = _is_strong_for_harvest(verdict, run_cfg)

        if debug:
            anchors = getattr(verdict, "anchors", None)
            n_anchors = 0 if not anchors else len(anchors)
            print(
                f"[moltie.debug] win={start}:{end} "
                f"relevant={verdict.relevant} "
                f"precedent_score={verdict.precedent_score} "
                f"confidence={verdict.confidence} "
                f"anchors={n_anchors} anchors_ok={ok} strong={strong}"
            )

        trace.append(
            {
                "phase": "harvest",
                "start": start,
                "end": end,
                "verdict": verdict.to_dict(),
                "anchors_ok": ok,
                "anchors_bad": bad[:6],
                # --- ADDED: include matched_X + evidence_to_x per-window for audit ---
                "matched_X": list(getattr(verdict, "matched_X", None) or []),
                "evidence_to_x": _evidence_to_x_from_verdict(doc_id, verdict),
            }
        )

        if not ok:
            continue

        if strong:
            harvested_verdicts.append(verdict)
            harvested_windows.append({"start": start, "end": end})
            for a in verdict.anchors or []:
                pid = a.para_id if hasattr(a, "para_id") else a.get("para_id")
                if pid:
                    harvested_para_ids.add(pid)

            # --- ADDED: accumulate evidence_to_x for strong windows only ---
            m = _evidence_to_x_from_verdict(doc_id, verdict)
            for evid, xs in m.items():
                harvested_evidence_to_x.setdefault(evid, set()).update(xs)

    if not harvested_verdicts or not harvested_para_ids:
        if debug:
            print(
                f"[moltie.loop] HARVEST EXIT no_signal "
                f"harvested_verdicts={len(harvested_verdicts)} "
                f"harvested_para_ids={len(harvested_para_ids)}"
            )

        nx = NegativeExit.from_best(
            atom_id=atom.atom_id,
            reason="no_signal",
            best=None,
            note=(
                f"No strong anchored fragments found in sweep; "
                f"n_paras={n} window_size={window_size} stride={stride} windows_scanned={total_windows}"
            ),

        )

        return LoopResult(verdict=None, negative_exit=nx, iters=1, trace=trace)

    harvested_paras_ordered = [
        {"para_id": pid, "text": full_para_map[pid]}
        for pid in full_para_ids_in_order
        if pid in harvested_para_ids
    ][:memory_max_items]

    evidence_pack2 = {
        "doc_id": doc_id,
        "doc_meta": {},
        "paras": harvested_paras_ordered,
        "retrieval": {"method": "harvest_memory"},
    }

    prompt2 = build_verifier_prompt(atom=atom, evidence_pack=evidence_pack2, cfg=run_cfg)
    verdict2 = verify_with_ollama(prompt2, llm_cfg)

    mem_para_map = {p["para_id"]: p["text"] for p in harvested_paras_ordered}
    ok2, bad2 = _anchors_valid(verdict2, mem_para_map)

    if debug:
        print(
            f"[moltie.debug] PASS-2 verdict: relevant={verdict2.relevant} "
            f"precedent_score={verdict2.precedent_score} "
            f"confidence={verdict2.confidence} "
            f"anchors={len(verdict2.anchors or [])} anchors_ok={ok2}"
        )

    if not ok2:
        best = sorted(
            harvested_verdicts,
            key=lambda v: (_nn_int(v.precedent_score), _nn_int(v.confidence)),
            reverse=True,
        )[0]

        return LoopResult(verdict=best, negative_exit=None, iters=1, trace=trace)

    # --- ADDED: attach consolidated evidence_to_x into trace tail for visibility ---
    # This does NOT change return schema; it's only for debug/audit.
    if harvested_evidence_to_x:
        trace.append(
            {
                "phase": "harvest_summary",
                "doc_id": doc_id,
                "atom_id": atom.atom_id,
                "evidence_to_x": {k: sorted(v) for k, v in harvested_evidence_to_x.items()},
                "n_evidence_items": len(harvested_evidence_to_x),
            }
        )

    return LoopResult(verdict=verdict2, negative_exit=None, iters=1, trace=trace)

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

    # --- E2E GUARD: fix the 'n_paras=1 huge blob' PDF extraction failure mode ---
    paras = _maybe_rechunk_single_blob_paras(paras)

    # --- NEW: harvest mode shortcut (keeps old behavior intact) ---
    if bool(getattr(run_cfg, "harvest_mode", False)):
        return run_harvest_then_reason_on_one_doc(doc_id, paras, atom, run_cfg, llm_cfg)

    trace: List[Dict[str, Any]] = []
    best: Optional[Verdict] = None
    best_quality = -1
    best_score = -1
    plateau = 0

    visited_starts: Set[int] = set()

    print(f"[moltie.loop] start doc_id={doc_id!r} atom_id={atom.atom_id!r} n_paras={len(paras)}")

    window_size = getattr(run_cfg, "window_size", 24)
    stride = getattr(run_cfg, "stride", 12)
    top_windows = getattr(run_cfg, "top_windows", 3)
    min_hits = getattr(run_cfg, "min_hits", 2)

    # include the final window start (n - window_size) even if stride doesn't land on it
    n = len(paras)
    starts = list(range(0, max(1, n), max(1, stride)))
    if n > window_size:
        last_start = max(0, n - window_size)
        if last_start not in starts:
            starts.append(last_start)
    starts = sorted(set(starts))
    est_total_windows = max(1, len(starts))

    # --- ADDED: evidence_to_x accumulator (para-level evidence keys) ---
    # evidence_id format: f"{doc_id}:{para_id}"
    evidence_to_x_all: Dict[str, Set[str]] = {}

    # --- ADDED: local helper to bind anchors -> matched_X (no external deps) ---
    def _evidence_to_x_from_verdict(_doc_id: str, _verdict: "Verdict") -> Dict[str, List[str]]:
        xs = [str(x).strip() for x in (getattr(_verdict, "matched_X", None) or []) if str(x).strip()]
        anchors = getattr(_verdict, "anchors", None) or []
        out: Dict[str, Set[str]] = {}
        for a in anchors:
            pid = a.para_id if hasattr(a, "para_id") else a.get("para_id")
            if not pid:
                continue
            evid = f"{_doc_id}:{str(pid).strip()}"
            out.setdefault(evid, set()).update(xs)
        return {k: sorted(v) for k, v in out.items()}

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

        picked = rr.retrieval.get("picked_windows") or []
        for w in picked:
            s = w.get("start")
            if isinstance(s, int):
                visited_starts.add(s)

        evidence_pack = {"doc_id": rr.doc_id, "doc_meta": {}, "paras": rr.paras, "retrieval": rr.retrieval}
        prompt = build_verifier_prompt(atom=atom, evidence_pack=evidence_pack, cfg=run_cfg)

        if atom.atom_id not in prompt:
            print(f"[moltie.loop] iter={i} WARN atom_id not found in prompt prompt_hash={_h(prompt)}")

        verdict = verify_with_ollama(prompt, llm_cfg)

        if not getattr(verdict, "atom_id", "").strip():
            print(f"[moltie.loop] iter={i} BAD_VERDICT missing atom_id prompt_hash={_h(prompt)} verdict={verdict.to_dict()}")
            raise RuntimeError("LLM returned Verdict without atom_id")

        # Anchor verification: verbatim + para_id
        para_map = {p["para_id"]: p["text"] for p in evidence_pack["paras"]}
        ok, bad = _anchors_valid(verdict, para_map)

        if not ok:
            trace.append(
                {
                    "iter": i,
                    "retrieval": rr.retrieval,
                    "verdict": verdict.to_dict(),
                    "error": {"type": "invalid_anchors", "details": bad[:6]},
                    # --- ADDED: include matched_X + evidence_to_x even for failures (useful for diagnosing) ---
                    "matched_X": list(getattr(verdict, "matched_X", None) or []),
                    "evidence_to_x": _evidence_to_x_from_verdict(doc_id, verdict),
                }
            )
            print(f"[moltie.loop] INVALID_ANCHORS iter={i} bad={bad[:3]} -> continuing")

            plateau += 1
            atom = refine_query(atom, verdict, i).atom
            continue

        # --- ADDED: per-iter evidence_to_x mapping (anchored only) ---
        evidence_to_x_step = _evidence_to_x_from_verdict(doc_id, verdict)
        for evid, xs in evidence_to_x_step.items():
            evidence_to_x_all.setdefault(evid, set()).update(xs)

        trace.append(
            {
                "iter": i,
                "retrieval": rr.retrieval,
                "verdict": verdict.to_dict(),
                # --- ADDED: audit visibility ---
                "matched_X": list(getattr(verdict, "matched_X", None) or []),
                "evidence_to_x": evidence_to_x_step,
            }
        )

        q = _compute_quality(verdict)
        print(
            f"[moltie.loop] iter={i} rel={getattr(verdict,'relevant',None)} "
            f"score={getattr(verdict,'precedent_score',None)} conf={getattr(verdict,'confidence',None)} "
            f"anchors={len(getattr(verdict,'anchors',[]) or [])} quality={q}"
        )

        if best is None or q >= (best_quality + int(run_cfg.eps_improve)):
            best, best_quality = verdict, q
            best_score = max(best_score, _nn_int(getattr(verdict, "precedent_score", 0)))
            plateau = 0
        else:
            plateau += 1

        conf = verdict.confidence
        if 0 <= run_cfg.thresh_conf <= 1:
            conf = verdict.confidence / 100.0

        anchors = getattr(verdict, "anchors", None) or []
        anchors_len = len(anchors)

        if (
            verdict.relevant
            and verdict.precedent_score >= run_cfg.thresh_score
            and conf >= run_cfg.thresh_conf
            and anchors_len >= run_cfg.anchors_required
        ):
            print(
                f"[moltie.loop] ACCEPT iter={i} score={verdict.precedent_score} "
                f"conf={verdict.confidence} anchors={anchors_len}"
            )

            # --- ADDED: attach consolidated evidence_to_x into trace tail for visibility ---
            if evidence_to_x_all:
                trace.append(
                    {
                        "phase": "evidence_to_x_summary",
                        "doc_id": doc_id,
                        "atom_id": atom.atom_id,
                        "evidence_to_x": {k: sorted(v) for k, v in evidence_to_x_all.items()},
                        "n_evidence_items": len(evidence_to_x_all),
                    }
                )

            return LoopResult(verdict=verdict, negative_exit=None, iters=i, trace=trace)

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

            # --- ADDED: attach consolidated evidence_to_x into trace tail for visibility ---
            if evidence_to_x_all:
                trace.append(
                    {
                        "phase": "evidence_to_x_summary",
                        "doc_id": doc_id,
                        "atom_id": atom.atom_id,
                        "evidence_to_x": {k: sorted(v) for k, v in evidence_to_x_all.items()},
                        "n_evidence_items": len(evidence_to_x_all),
                    }
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

    # --- ADDED: attach consolidated evidence_to_x into trace tail for visibility ---
    if evidence_to_x_all:
        trace.append(
            {
                "phase": "evidence_to_x_summary",
                "doc_id": doc_id,
                "atom_id": atom.atom_id,
                "evidence_to_x": {k: sorted(v) for k, v in evidence_to_x_all.items()},
                "n_evidence_items": len(evidence_to_x_all),
            }
        )

    return LoopResult(verdict=None, negative_exit=nx, iters=run_cfg.max_iters, trace=trace)
