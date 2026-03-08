from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm
import re


# =========================================================
# CONFIG
# =========================================================

@dataclass(frozen=True)
class NeedleExplorerConfig:
    model: str = "mistral-small3.2:latest"
    ollama_cmd: str = "ollama"
    temperature: float = 0.2
    timeout_sec: int = 180

    # extraction controls
    max_needles_per_text: int = 8
    min_confidence: float = 0.50

    # consolidation controls
    max_promoted_needles: int = 50
    min_support_count: int = 2

    # io
    text_id_col: str = "text_id"
    text_col: str = "text"


# =========================================================
# OLLAMA CALL
# =========================================================

def _run_ollama_json(prompt: str, cfg: NeedleRegexBuilderConfig) -> Dict[str, Any]:
    cmd = [cfg.ollama_cmd, "run", cfg.model]

    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        timeout=cfg.timeout_sec,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Ollama failed with code {proc.returncode}\nSTDERR:\n{proc.stderr[:1000]}"
        )

    raw = (proc.stdout or "").strip()

    # -----------------------------------------------------
    # 1) direct parse
    # -----------------------------------------------------
    try:
        return json.loads(raw)
    except Exception:
        pass

    # -----------------------------------------------------
    # 2) strip markdown code fences if present
    # -----------------------------------------------------
    cleaned = raw.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        # drop first fence
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        # drop final fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned = "\n".join(lines).strip()

    # -----------------------------------------------------
    # 3) extract outer JSON object
    # -----------------------------------------------------
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start:end + 1].strip()

    # -----------------------------------------------------
    # 4) convert Python raw triple-quoted regex strings
    #    into valid JSON strings
    #
    #    Example:
    #    "regex_pattern": r""" abc """
    #    becomes:
    #    "regex_pattern": "abc"
    # -----------------------------------------------------
    def _replace_raw_triple_quote(match: re.Match) -> str:
        inner = match.group(1)

        # escape backslashes, quotes, and newlines for JSON
        inner_json = (
            inner
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\r", "")
            .replace("\n", "\\n")
        )

        return f'"regex_pattern": "{inner_json}"'

    cleaned = re.sub(
        r'"regex_pattern"\s*:\s*r?"""(.*?)"""',
        _replace_raw_triple_quote,
        cleaned,
        flags=re.DOTALL,
    )

    # also handle single-quoted raw strings if model ever emits them
    def _replace_raw_single_quote(match: re.Match) -> str:
        inner = match.group(1)
        inner_json = (
            inner
            .replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\r", "")
            .replace("\n", "\\n")
        )
        return f'"regex_pattern": "{inner_json}"'

    cleaned = re.sub(
        r'"regex_pattern"\s*:\s*r?"(.*?)"',
        _replace_raw_single_quote,
        cleaned,
        flags=re.DOTALL,
    )

    # -----------------------------------------------------
    # 5) retry parse
    # -----------------------------------------------------
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    raise ValueError(f"Model did not return valid JSON. Raw output:\n{raw[:2000]}")


# =========================================================
# PROMPTS
# =========================================================

def _build_extraction_prompt(text_id: str, text: str, cfg: NeedleExplorerConfig) -> str:
    return f"""
You are extracting reusable LEGAL / ARGUMENTATIVE NEEDLES from text.

Your job is NOT to summarise the text.
Your job is NOT to retell the story.
Your job is to identify a SMALL number of reusable legal or procedural issue-labels
that could later be used to match this text against employment tribunal or appeal cases.

A "needle" is a stable issue-family label, not a case-specific description.

GOOD needle labels:
- failure to investigate duties
- assumed intention
- no contemporaneous record
- predetermination
- inconsistent reasoning
- appeal scope limitation
- lack of prior warning
- role expectation mismatch

BAD needle labels:
- role evolution not reflected in assessment
- management endorsed problematic practices
- disciplinary action based on unchanged conduct
- reliance on management's silence
- Leo was dismissed in April
- manager said X in meeting
- Bloomberg investigation issue

IMPORTANT:
Prefer ABSTRACT, REUSABLE issue labels over detailed narrative phrasing.
If several possible labels mean roughly the same thing, choose the SHORTEST and most reusable one.

LABEL RULES:
- labels must be lowercase
- labels must be 2 to 5 words only
- labels must describe a reusable legal / procedural / evidential issue
- labels must NOT describe a sequence of events
- labels must NOT mention names, employers, dates, or case-specific trivia
- labels should sound like categories, not conclusions
- avoid labels containing phrases like:
  "based on", "led to", "not reflected in", "reliance on", "failure to consider",
  "documented change in", "treatment of", "application of", "change in role"
- prefer canonical motifs such as:
  investigation failure, role expectation mismatch, inconsistent reasoning,
  lack of warning, no contemporaneous record, predetermination, appeal scope limitation,
  assumed intention, comparator inconsistency, disproportionate sanction

MERGING RULE:
If the text supports a more specific phrase and a broader reusable phrase,
return ONLY the broader reusable phrase.

For example:
- "role evolution not reflected in assessment" -> "role expectation mismatch"
- "documented change in role expectations" -> "role expectation mismatch"
- "lack of contemporaneous instruction or guidance" -> "lack of contemporaneous guidance"
- "management failed to warn of risks" -> "lack of warning"

Return STRICT JSON only with this shape:

{{
  "text_id": "{text_id}",
  "candidate_needles": [
    {{
      "label": "short reusable needle",
      "why_it_matters": "one sentence",
      "evidence_quote": "short quote from text",
      "confidence": 0.0
    }}
  ]
}}

Rules:
- produce at most {cfg.max_needles_per_text} needles
- confidence must be between 0 and 1
- prefer fewer strong needles over many weak ones
- avoid duplicate meanings
- if no strong needles exist, return an empty list
- do not output any text before or after the JSON

TEXT:
\"\"\"
{text}
\"\"\"
""".strip()

def _build_consolidation_prompt(all_candidates: List[Dict[str, Any]], cfg: NeedleExplorerConfig) -> str:
    payload = json.dumps(all_candidates, ensure_ascii=False, indent=2)

    return f"""
You are consolidating exploratory legal needles into a promoted reusable set.

You will receive candidate needles extracted from many texts.
Many labels will overlap semantically.
Your job is to MERGE semantically similar labels into stable promoted needles.

Return STRICT JSON only:

{{
  "promoted_needles": [
    {{
      "needle": "stable promoted label",
      "aliases": ["similar label 1", "similar label 2"],
      "why_it_matters": "one sentence",
      "support_count": 0
    }}
  ]
}}

Rules:
- merge labels that clearly mean the same thing
- keep promoted labels concise, reusable, and legally meaningful
- do not force unrelated labels together
- preserve nuance where it matters
- do not exceed {cfg.max_promoted_needles} promoted needles
- support_count should reflect how many candidate items support the promoted needle
- only include promoted needles that are actually supported by the input

CANDIDATES:
{payload}
""".strip()


# =========================================================
# EXTRACTION
# =========================================================

def extract_needles_from_text(
    text: str,
    text_id: str,
    cfg: NeedleExplorerConfig,
) -> Dict[str, Any]:
    prompt = _build_extraction_prompt(text_id=text_id, text=text, cfg=cfg)
    out = _run_ollama_json(prompt, cfg)

    candidates = out.get("candidate_needles", [])
    if not isinstance(candidates, list):
        candidates = []

    cleaned: List[Dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue

        label = str(item.get("label", "")).strip().lower()
        why = str(item.get("why_it_matters", "")).strip()
        quote = str(item.get("evidence_quote", "")).strip()

        try:
            conf = float(item.get("confidence", 0.0))
        except Exception:
            conf = 0.0

        if not label:
            continue
        if conf < cfg.min_confidence:
            continue

        cleaned.append(
            {
                "text_id": text_id,
                "label": label,
                "why_it_matters": why,
                "evidence_quote": quote,
                "confidence": conf,
            }
        )

    return {
        "text_id": text_id,
        "candidate_needles": cleaned,
    }


def extract_needles_from_dataframe(
    df: pd.DataFrame,
    cfg: NeedleExplorerConfig,
) -> pd.DataFrame:
    if cfg.text_col not in df.columns:
        raise KeyError(f"Missing text column: {cfg.text_col}")

    work = df.copy()

    if cfg.text_id_col not in work.columns:
        work[cfg.text_id_col] = [f"row_{i:04d}" for i in range(1, len(work) + 1)]

    rows: List[Dict[str, Any]] = []

    for _, r in tqdm(work.iterrows(), total=len(work), desc="Extracting needles", unit="row"):
        text_id = str(r[cfg.text_id_col])
        text = str(r[cfg.text_col])

        result = extract_needles_from_text(text=text, text_id=text_id, cfg=cfg)
        for item in result["candidate_needles"]:
            rows.append(item)

    if not rows:
        return pd.DataFrame(
            columns=["text_id", "label", "why_it_matters", "evidence_quote", "confidence"]
        )

    out_df = pd.DataFrame(rows).sort_values(
        ["text_id", "confidence"], ascending=[True, False]
    ).reset_index(drop=True)

    return out_df


# =========================================================
# CONSOLIDATION
# =========================================================

def consolidate_needles(
    candidate_df: pd.DataFrame,
    cfg: NeedleExplorerConfig,
) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame(
            columns=["needle", "aliases", "why_it_matters", "support_count"]
        )

    records = candidate_df[
        ["text_id", "label", "why_it_matters", "confidence"]
    ].to_dict(orient="records")

    prompt = _build_consolidation_prompt(records, cfg)
    out = _run_ollama_json(prompt, cfg)

    promoted = out.get("promoted_needles", [])
    if not isinstance(promoted, list):
        promoted = []

    rows: List[Dict[str, Any]] = []
    for item in promoted:
        if not isinstance(item, dict):
            continue

        needle = str(item.get("needle", "")).strip().lower()
        aliases = item.get("aliases", [])
        why = str(item.get("why_it_matters", "")).strip()

        try:
            support_count = int(item.get("support_count", 0))
        except Exception:
            support_count = 0

        if not needle:
            continue
        if support_count < cfg.min_support_count:
            continue

        if not isinstance(aliases, list):
            aliases = []

        aliases = [str(x).strip().lower() for x in aliases if str(x).strip()]

        rows.append(
            {
                "needle": needle,
                "aliases": aliases,
                "why_it_matters": why,
                "support_count": support_count,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["needle", "aliases", "why_it_matters", "support_count"]
        )

    return pd.DataFrame(rows).sort_values(
        ["support_count", "needle"], ascending=[False, True]
    ).reset_index(drop=True)


# =========================================================
# ORCHESTRATOR
# =========================================================

def run_needle_explorer(
    input_df: pd.DataFrame,
    cfg: Optional[NeedleExplorerConfig] = None,
) -> Dict[str, pd.DataFrame]:
    cfg = cfg or NeedleExplorerConfig()

    candidate_df = extract_needles_from_dataframe(input_df, cfg)
    promoted_df = consolidate_needles(candidate_df, cfg)

    return {
        "candidate_needles_df": candidate_df,
        "promoted_needles_df": promoted_df,
    }


# =========================================================
# OPTIONAL FILE RUNNER
# =========================================================

def run_needle_explorer_from_csv(
    csv_path: Path,
    text_col: str,
    text_id_col: Optional[str] = None,
    cfg: Optional[NeedleExplorerConfig] = None,
) -> Dict[str, pd.DataFrame]:
    cfg = cfg or NeedleExplorerConfig()
    df = pd.read_csv(csv_path)

    if text_id_col is not None:
        cfg = NeedleExplorerConfig(
            model=cfg.model,
            ollama_cmd=cfg.ollama_cmd,
            temperature=cfg.temperature,
            timeout_sec=cfg.timeout_sec,
            max_needles_per_text=cfg.max_needles_per_text,
            min_confidence=cfg.min_confidence,
            max_promoted_needles=cfg.max_promoted_needles,
            min_support_count=cfg.min_support_count,
            text_id_col=text_id_col,
            text_col=text_col,
        )
    else:
        cfg = NeedleExplorerConfig(
            model=cfg.model,
            ollama_cmd=cfg.ollama_cmd,
            temperature=cfg.temperature,
            timeout_sec=cfg.timeout_sec,
            max_needles_per_text=cfg.max_needles_per_text,
            min_confidence=cfg.min_confidence,
            max_promoted_needles=cfg.max_promoted_needles,
            min_support_count=cfg.min_support_count,
            text_id_col=cfg.text_id_col,
            text_col=text_col,
        )

    return run_needle_explorer(df, cfg)


if __name__ == "__main__":
    # tiny demo
    demo_df = pd.DataFrame(
        [
            {
                "text_id": "row_0001",
                "text": (
                    "The employer concluded he was avoiding tickets, but did not investigate "
                    "whether ticket handling was actually part of his primary senior duties. "
                    "There was no contemporaneous note of the July 2023 meeting."
                ),
            },
            {
                "text_id": "row_0002",
                "text": (
                    "The appeal rejected the point as outside scope and failed to engage with "
                    "the actual complaint. The outcome appeared predetermined."
                ),
            },
        ]
    )

    cfg = NeedleExplorerConfig(model="mistral-small3.2:latest")
    out = run_needle_explorer(demo_df, cfg)

    print("\n=== CANDIDATE NEEDLES ===")
    print(out["candidate_needles_df"].to_string(index=False))

    print("\n=== PROMOTED NEEDLES ===")
    print(out["promoted_needles_df"].to_string(index=False))