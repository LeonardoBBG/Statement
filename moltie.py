# moltie_single_doc_harness.py
# ==========================================================
# MOLTIE SINGLE-DOC RUN HARNESS (IMPORTABLE MODULE)
#  - no fake evidence_to_x
#  - deterministic client_cfg pin (num_predict + stop + timeout)
#  - single DEBUG switch controls verbosity + forensic prints
#  - notebook supplies: pdf_path, paras, y_path, and config overrides
# ==========================================================

from __future__ import annotations

import sys
import json
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm


# -------------------------
# Public config containers
# -------------------------

@dataclass(frozen=True)
class IterTempSchedule:
    enabled: bool = True
    start: float = 0.20
    end: float = 0.00
    curve: str = "linear"   # "linear" or "exp"
    cap: float = 0.80

    def to_runconfig_dict(self) -> Dict[str, Any]:
        return {
            "iter_temp_enabled": self.enabled,
            "iter_temp_start": self.start,
            "iter_temp_end": self.end,
            "iter_temp_curve": self.curve,
            "iter_temp_cap": self.cap,
        }


@dataclass(frozen=True)
class HarnessPaths:
    repo_root: Path = Path("/home/hello/Projects/Statements")
    code_root: Optional[Path] = None  # defaults to repo_root/"code"

    def resolved(self) -> "HarnessPaths":
        rr = self.repo_root.resolve()
        cr = (self.code_root or (rr / "code")).resolve()
        return HarnessPaths(repo_root=rr, code_root=cr)


@dataclass
class HarnessDefaults:
    # Client defaults (pin to avoid truncation + state leaks)
    client_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "model": "mistral-small3.2:latest",
        "ollama_url": "http://localhost:11434/api/generate",
        "timeout_s": 180,
        "temperature": 0.0,  # fallback if iter_temp_enabled=False
        "num_predict": 1000,
        "max_retries": 2,
    })

    # RunConfig defaults (can be overridden per call)
    runconfig_base: Dict[str, Any] = field(default_factory=lambda: {
        "harvest_mode": False,
        "max_iters": 3,
        "window_size": 12,
        "stride": 6,
        "top_windows": 3,
        "k_chunks_per_doc": 12,
        "anchors_required": 1,
        "min_hits": 1,
        "thresh_score": 1,
        "thresh_conf": 0.8,   # loop normalizes 0..1
        "plateau_p": 2,
        "eps_improve": 0,
    })


# -------------------------
# Core runner
# -------------------------

def run_moltie_single_doc(
    *,
    pdf_path: Path,
    paras: List[Dict[str, Any]],
    y_path: Path,
    paths: HarnessPaths = HarnessPaths(),
    defaults: HarnessDefaults = HarnessDefaults(),
    runconfig_overrides: Optional[Dict[str, Any]] = None,
    client_overrides: Optional[Dict[str, Any]] = None,
    iter_temp: IterTempSchedule = IterTempSchedule(),
    debug: bool = True,
    x_keys: Optional[Sequence[str]] = None,
    reload_modules: bool = True,
) -> List[Tuple[str, Any]]:
    """
    Importable runner:
      - pdf_path: Path to the PDF for DOC_ID
      - paras: list[{"para_id":..., "text":...}, ...] (non-empty)
      - y_path: Path to Y_inferred.json (or equivalent)
      - runconfig_overrides: per-notebook overrides (merged on top of defaults.runconfig_base)
      - client_overrides: per-notebook overrides (merged on top of defaults.client_kwargs)
      - iter_temp: schedule injected into RunConfig (if loop.py supports it)
      - x_keys: optionally restrict which x_tests to run (otherwise runs all sorted keys)
      - returns: list of (x_key, res)
    """

    def dprint(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    # -------------------------
    # 0) Resolve + validate repo anchors
    # -------------------------
    paths = paths.resolved()
    repo_root = paths.repo_root
    code_root = paths.code_root

    assert repo_root.exists(), f"Missing repo root: {repo_root}"
    assert code_root.exists(), f"Missing code root: {code_root}"

    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    # -------------------------
    # 1) Import modules (and optionally reload)
    # -------------------------
    import moltie.schemas.run_config as rc_mod
    import moltie.llm.verifier_prompt as vp_mod
    import moltie.llm.client as client_mod
    import moltie.schemas.query_object as qo_mod
    import moltie.agent.loop as loop_mod

    if reload_modules:
        importlib.reload(rc_mod)      # config first
        importlib.reload(vp_mod)      # prompt
        importlib.reload(client_mod)  # client boundary
        importlib.reload(qo_mod)      # AtomQuery helpers
        importlib.reload(loop_mod)    # loop last

    RunConfig = rc_mod.RunConfig
    AtomQuery = qo_mod.AtomQuery
    run_agent_on_one_doc = loop_mod.run_agent_on_one_doc
    LLMClientConfig = client_mod.LLMClientConfig

    # -------------------------
    # 2) Validate inputs
    # -------------------------
    assert isinstance(pdf_path, Path), "pdf_path must be a pathlib.Path"
    assert pdf_path.exists(), f"Missing PDF file: {pdf_path}"
    assert isinstance(paras, list) and len(paras) > 0, "paras must be non-empty list"
    assert isinstance(paras[0], dict) and "para_id" in paras[0] and "text" in paras[0], \
        "paras must be list of dicts with para_id/text"

    # Ensure loop rechunking consistency
    paras = loop_mod._maybe_rechunk_single_blob_paras(paras)

    doc_id = pdf_path.stem

    dprint("[moltie.debug] repo:", repo_root)
    dprint("[moltie.debug] code:", code_root)
    dprint("[moltie.debug] loop file:", Path(loop_mod.__file__).resolve())
    dprint("[moltie.debug] doc_id:", doc_id)
    dprint("[moltie.debug] paras:", len(paras),
           "sample:", paras[0]["para_id"],
           "len:", len(paras[0].get("text") or ""))

    # -------------------------
    # 2b) Pin LLM client cfg (override-friendly)
    # -------------------------
    _client_kwargs = dict(defaults.client_kwargs)
    if client_overrides:
        _client_kwargs.update(client_overrides)

    # optional stop pin (only if supported by dataclass)
    try:
        if "stop" in getattr(LLMClientConfig, "__annotations__", {}):
            _client_kwargs.setdefault("stop", [])
    except Exception:
        pass

    # debug pin (only if supported by dataclass)
    try:
        if "debug" in getattr(LLMClientConfig, "__annotations__", {}):
            _client_kwargs["debug"] = debug
    except Exception:
        pass

    client_cfg = LLMClientConfig(**_client_kwargs)

    dprint("[moltie.debug] client_cfg.model:", getattr(client_cfg, "model", None))
    dprint("[moltie.debug] client_cfg.num_predict:", getattr(client_cfg, "num_predict", None))
    dprint("[moltie.debug] client_cfg.timeout_s:", getattr(client_cfg, "timeout_s", None))
    dprint("[moltie.debug] client_cfg.temperature(fallback):", getattr(client_cfg, "temperature", None))
    if hasattr(client_cfg, "stop"):
        dprint("[moltie.debug] client_cfg.stop:", getattr(client_cfg, "stop", None))
    if hasattr(client_cfg, "debug"):
        dprint("[moltie.debug] client_cfg.debug:", getattr(client_cfg, "debug", None))

    # -------------------------
    # 3) Load Y JSON
    # -------------------------
    assert isinstance(y_path, Path), "y_path must be a pathlib.Path"
    assert y_path.exists(), f"Missing Y file: {y_path}"
    y_json = json.loads(y_path.read_text(encoding="utf-8"))

    x_tests_map = (y_json.get("x_tests") or {})
    assert isinstance(x_tests_map, dict) and len(x_tests_map) > 0, "Y JSON has no x_tests"

    dprint("[moltie.debug] Y file:", y_path)
    dprint("[moltie.debug] Y x_tests:", len(x_tests_map))

    # -------------------------
    # 4) Determine X keys to run
    # -------------------------
    all_keys = sorted(x_tests_map.keys())
    if x_keys is None:
        run_keys = all_keys
    else:
        # Keep notebook order if provided, but validate membership
        missing = [k for k in x_keys if k not in x_tests_map]
        assert not missing, f"x_keys contains unknown keys: {missing}"
        run_keys = list(x_keys)

    dprint("[moltie.debug] iter_temp_enabled:", iter_temp.enabled)
    if iter_temp.enabled:
        dprint("[moltie.debug] iter_temp_start/end/curve/cap:",
               iter_temp.start, iter_temp.end, iter_temp.curve, iter_temp.cap)

    # tqdm bar only when debug=False
    pbar = tqdm(run_keys, desc=f"moltie | doc={doc_id}", unit="atom") if not debug else run_keys

    results: List[Tuple[str, Any]] = []

    for x_key in pbar:
        x_name = (x_tests_map.get(x_key) or {}).get("name", x_key)

        if not debug:
            pbar.set_postfix_str(f"{x_key} | {x_name}")

        dprint("\n==================================================")
        dprint("Running atom:", x_key, "->", x_name)
        dprint("==================================================")

        merged = qo_mod.merge_indicators_and_excludes(y_json, [x_key])

        atom = AtomQuery(
            atom_id=x_key,
            x_tests=[x_key],
            proposition=x_name,
            positive_indicators=merged["positive_indicators"],
            excludes=merged["excludes"],
            keyword_seeds=merged["positive_indicators"],
            expansion_terms=[],
        )

        # Build RunConfig dict deterministically (defaults -> overrides -> temp schedule -> debug)
        cfg_dict = dict(defaults.runconfig_base)
        if runconfig_overrides:
            cfg_dict.update(runconfig_overrides)
        cfg_dict.update(iter_temp.to_runconfig_dict())
        cfg_dict["debug"] = debug

        cfg2 = RunConfig.from_dict(cfg_dict)

        res = run_agent_on_one_doc(
            doc_id,
            paras,
            atom,
            cfg2,
            client_cfg,
        )

        results.append((x_key, res))

        # -------------------------
        # 4b) Per-atom output + FORENSICS
        # -------------------------
        dprint("\nRESULT for", x_key)
        if getattr(res, "verdict", None):
            v = res.verdict.to_dict()
            dprint(
                "relevant=", v.get("relevant"),
                "score=", v.get("precedent_score"),
                "conf=", v.get("confidence"),
                "anchors=", len(v.get("anchors") or []),
            )
            dprint("anchors:")
            if debug:
                pprint(v.get("anchors"))
        else:
            neg = getattr(res, "negative_exit", None)
            dprint("NEGATIVE:", getattr(neg, "reason", None) if neg else None)

            # FORENSIC: find FIRST invalid_anchors and print quote vs paragraph text
            found = False
            for t in (getattr(res, "trace", None) or []):
                err = (t.get("error") or {})
                if err.get("type") != "invalid_anchors":
                    continue

                found = True
                dprint("\n[FORENSIC] iter:", t.get("iter"))
                dprint("[FORENSIC] bad_details:", err.get("details"))

                v = t.get("verdict") or {}
                anchors = v.get("anchors") or []
                dprint("[FORENSIC] anchors_from_model_count:", len(anchors))
                if debug:
                    pprint(anchors[:3])

                bad_details = err.get("details") or []
                for typ, pid in bad_details[:3]:
                    dprint(f"\n[FORENSIC] bad_type={typ} pid={pid}")

                    a_hits = [a for a in anchors if isinstance(a, dict) and a.get("para_id") == pid]
                    if not a_hits:
                        dprint("[FORENSIC] model_anchor_for_pid: NONE (pid mismatch or missing)")
                    else:
                        q0 = (a_hits[0].get("quote") or "")
                        dprint("[FORENSIC] model_quote_repr:", repr(q0)[:1200])

                    ptxt = next((p.get("text", "") for p in paras if p.get("para_id") == pid), "")
                    dprint("[FORENSIC] global_para_text_len:", len(ptxt))
                    dprint("[FORENSIC] global_para_text_repr:", repr(ptxt)[:1600])

                    if a_hits:
                        q0 = (a_hits[0].get("quote") or "")
                        dprint("[FORENSIC] raw_contains:", (q0 in ptxt))

                break

            if not found:
                dprint("[FORENSIC] no invalid_anchors entries in trace (different failure mode).")

    # -------------------------
    # 5) Compact end summary (keeps your original vibe)
    # -------------------------
    dprint("\n================ RESULT ================")
    last_res = results[-1][1] if results else None
    if last_res and getattr(last_res, "verdict", None):
        v = last_res.verdict.to_dict()
        dprint(
            "VERDICT: relevant=", v.get("relevant"),
            "score=", v.get("precedent_score"),
            "conf=", v.get("confidence"),
            "anchors=", len(v.get("anchors") or []),
            "matched_X=", v.get("matched_X"),
        )
        dprint("\nanchors:")
        if debug:
            pprint(v.get("anchors"))
    else:
        dprint("NEGATIVE_EXIT:")
        if debug and last_res and getattr(last_res, "negative_exit", None):
            pprint(last_res.negative_exit.to_dict())
        elif debug:
            pprint(None)

    dprint("\n================ TRACE (last 3 entries) ================")
    if debug:
        pprint((getattr(last_res, "trace", None) or [])[-3:] if last_res else None)

    dprint("\n================ TRACE SUMMARY ================")
    dprint(
        "iters:", getattr(last_res, "iters", None) if last_res else None,
        "trace_len:", len(getattr(last_res, "trace", None) or []) if last_res else None,
    )

    print("\n================ END OF RUN ================")
    return results


# -------------------------
# Optional tiny helper for notebooks
# -------------------------

def load_y_json(y_path: Path) -> Dict[str, Any]:
    assert isinstance(y_path, Path), "y_path must be a pathlib.Path"
    assert y_path.exists(), f"Missing Y file: {y_path}"
    return json.loads(y_path.read_text(encoding="utf-8"))