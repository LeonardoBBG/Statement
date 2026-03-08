from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern

import pandas as pd
from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

@dataclass(frozen=True)
class NeedleRegexBuilderConfig:
    model: str = "mistral-small3.2:latest"
    ollama_cmd: str = "ollama"
    timeout_sec: int = 180

    # input columns
    needle_col: str = "label"
    quote_col: str = "evidence_quote"
    text_id_col: str = "text_id"
    text_col: Optional[str] = None   # optional, can enrich regex generation

    # generation controls
    min_quotes_per_needle: int = 1
    max_quotes_per_needle: int = 12
    max_texts_per_needle: int = 6
    min_regex_confidence: float = 0.50

    # validation controls
    validation_min_hit_rate: float = 0.50
    validation_min_hits: int = 1

    # output / regex controls
    regex_flags: List[str] = None

    def __post_init__(self):
        if self.regex_flags is None:
            object.__setattr__(self, "regex_flags", ["IGNORECASE", "VERBOSE"])


# =========================================================
# OLLAMA JSON HELPER
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
    # helper: escape arbitrary text into JSON-safe string
    # -----------------------------------------------------
    def _json_escape(s: str) -> str:
        return (
            s.replace("\\", "\\\\")
             .replace('"', '\\"')
             .replace("\r", "")
             .replace("\n", "\\n")
             .replace("\t", "\\t")
        )

    # -----------------------------------------------------
    # helper: normalize markdown/codefence/python-string junk
    # -----------------------------------------------------
    def _clean_model_output(text: str) -> str:
        cleaned = text.strip()

        # 1) remove markdown fences
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            cleaned = "\n".join(lines).strip()

        # 2) isolate outer JSON object
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1].strip()

        # 3) convert regex_pattern: r""" ... """
        cleaned = re.sub(
            r'("regex_pattern"\s*:\s*)r?"""(.*?)"""',
            lambda m: m.group(1) + '"' + _json_escape(m.group(2)) + '"',
            cleaned,
            flags=re.DOTALL,
        )

        # 4) convert regex_pattern: r''' ... '''
        cleaned = re.sub(
            r'("regex_pattern"\s*:\s*)r?\'\'\'(.*?)\'\'\'',
            lambda m: m.group(1) + '"' + _json_escape(m.group(2)) + '"',
            cleaned,
            flags=re.DOTALL,
        )

        # 5) convert regex_pattern: r" ... "
        cleaned = re.sub(
            r'("regex_pattern"\s*:\s*)r"(.*?)"',
            lambda m: m.group(1) + '"' + _json_escape(m.group(2)) + '"',
            cleaned,
            flags=re.DOTALL,
        )

        # 6) convert regex_pattern: r' ... '
        cleaned = re.sub(
            r'("regex_pattern"\s*:\s*)r\'(.*?)\'',
            lambda m: m.group(1) + '"' + _json_escape(m.group(2)) + '"',
            cleaned,
            flags=re.DOTALL,
        )

        return cleaned

    # -----------------------------------------------------
    # 1) direct parse
    # -----------------------------------------------------
    try:
        return json.loads(raw)
    except Exception:
        pass

    # -----------------------------------------------------
    # 2) cleaned parse
    # -----------------------------------------------------
    cleaned = _clean_model_output(raw)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # -----------------------------------------------------
    # 3) debug visibility
    # -----------------------------------------------------
    raise ValueError(
        "Model did not return valid JSON.\n\n"
        f"RAW OUTPUT:\n{raw[:2000]}\n\n"
        f"CLEANED OUTPUT:\n{cleaned[:2000]}"
    )

# =========================================================
# REGEX FLAG HELPER
# =========================================================

def _compile_pattern(pattern: str, flags_list: List[str]) -> Pattern[str]:
    flags = 0
    for f in flags_list:
        fu = str(f).strip().upper()
        if fu == "IGNORECASE":
            flags |= re.IGNORECASE
        elif fu == "VERBOSE":
            flags |= re.VERBOSE
        elif fu == "MULTILINE":
            flags |= re.MULTILINE
        elif fu == "DOTALL":
            flags |= re.DOTALL
    return re.compile(pattern, flags)


# =========================================================
# PROMPTS
# =========================================================

def _build_regex_generation_prompt(
    needle: str,
    quotes: List[str],
    source_texts: Optional[List[str]],
    cfg: NeedleRegexBuilderConfig,
) -> str:
    quotes_payload = json.dumps(quotes, ensure_ascii=False, indent=2)
    texts_payload = json.dumps(source_texts or [], ensure_ascii=False, indent=2)

    return f"""
You are building a REGEX FAMILY from discovered legal-language evidence.

Your job:
Generate ONE regex pattern for the needle below.

CRITICAL RULES:
- Do NOT use any manual legal dictionary.
- Do NOT invent a taxonomy.
- Build the regex only from:
  1. the needle label
  2. the evidence quotes
  3. close linguistic variants naturally implied by those quotes
- The regex should be broad enough to catch close paraphrases,
  but not so broad that it matches generic legal language.
- Prefer phrase-family matching, not single exact literal phrases.
- Do NOT output a regex that simply matches every occurrence of the needle words literally.
- Do NOT add concepts not grounded in the evidence.
- The result must be self-evolving from the evidence.

NEEDLE:
"{needle}"

EVIDENCE QUOTES:
{quotes_payload}

OPTIONAL SOURCE TEXTS:
{texts_payload}

Return STRICT JSON only in this shape:

{{
  "needle": "{needle}",
  "regex_pattern": "regex here",
  "seed_phrases": ["phrase 1", "phrase 2"],
  "why_this_regex": "brief explanation",
  "confidence": 0.0
}}

REGEX REQUIREMENTS:
- regex_pattern must be a plain JSON string
- do NOT use markdown code fences
- do NOT use Python raw strings such as r"...", r'...', r\"\"\"...\"\"\"
- do NOT use triple quotes
- escape backslashes so the value is valid JSON
- valid Python regex once parsed from JSON
- designed for re.IGNORECASE | re.VERBOSE
- should usually contain grouped alternatives
- should capture phrase families, not only exact full quotes
- should avoid extremely generic words on their own
- confidence must be between 0 and 1

GOOD BEHAVIOUR:
- infer close textual variants from the evidence
- abstract repeated wording patterns
- preserve the core issue family

BAD BEHAVIOUR:
- inventing unrelated legal synonyms
- creating a regex so broad that it matches generic reasoning
- relying only on the needle label instead of the evidence
- outputting prose instead of JSON

Return JSON only.
""".strip()


# =========================================================
# VALIDATION
# =========================================================

def validate_regex_on_quotes(
    regex_pattern: str,
    quotes: List[str],
    flags_list: List[str],
) -> Dict[str, Any]:
    try:
        rx = _compile_pattern(regex_pattern, flags_list)
    except Exception as e:
        return {
            "regex_valid": False,
            "regex_error": f"{type(e).__name__}: {str(e)}",
            "n_quotes": len(quotes),
            "n_hits": 0,
            "hit_rate": 0.0,
            "matching_quotes": [],
        }

    matching_quotes: List[str] = []
    for q in quotes:
        q_str = str(q or "")
        if rx.search(q_str):
            matching_quotes.append(q_str)

    n_quotes = len(quotes)
    n_hits = len(matching_quotes)
    hit_rate = (n_hits / n_quotes) if n_quotes > 0 else 0.0

    return {
        "regex_valid": True,
        "regex_error": "",
        "n_quotes": n_quotes,
        "n_hits": n_hits,
        "hit_rate": hit_rate,
        "matching_quotes": matching_quotes,
    }


# =========================================================
# PER-NEEDLE REGEX GENERATION
# =========================================================

def build_regex_for_needle_group(
    needle: str,
    quotes: List[str],
    source_texts: Optional[List[str]],
    cfg: NeedleRegexBuilderConfig,
) -> Dict[str, Any]:
    clean_quotes = [str(x).strip() for x in quotes if str(x).strip()]
    clean_texts = [str(x).strip() for x in (source_texts or []) if str(x).strip()]

    if len(clean_quotes) < cfg.min_quotes_per_needle:
        return {
            "needle": needle,
            "regex_pattern": "",
            "seed_phrases": [],
            "why_this_regex": "Insufficient quote support.",
            "confidence": 0.0,
            "regex_valid": False,
            "regex_error": "Insufficient quote support.",
            "n_quotes": len(clean_quotes),
            "n_hits": 0,
            "hit_rate": 0.0,
            "matching_quotes": [],
            "accepted": False,
        }

    prompt = _build_regex_generation_prompt(
        needle=needle,
        quotes=clean_quotes[: cfg.max_quotes_per_needle],
        source_texts=clean_texts[: cfg.max_texts_per_needle],
        cfg=cfg,
    )

    out = _run_ollama_json(prompt, cfg)

    regex_pattern = str(out.get("regex_pattern", "")).strip()
    seed_phrases = out.get("seed_phrases", [])
    why_this_regex = str(out.get("why_this_regex", "")).strip()

    try:
        confidence = float(out.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    if not isinstance(seed_phrases, list):
        seed_phrases = []

    seed_phrases = [str(x).strip() for x in seed_phrases if str(x).strip()]

    validation = validate_regex_on_quotes(
        regex_pattern=regex_pattern,
        quotes=clean_quotes,
        flags_list=cfg.regex_flags,
    )

    accepted = bool(
        validation["regex_valid"]
        and confidence >= cfg.min_regex_confidence
        and validation["n_hits"] >= cfg.validation_min_hits
        and validation["hit_rate"] >= cfg.validation_min_hit_rate
    )

    return {
        "needle": needle,
        "regex_pattern": regex_pattern,
        "seed_phrases": seed_phrases,
        "why_this_regex": why_this_regex,
        "confidence": confidence,
        "regex_valid": validation["regex_valid"],
        "regex_error": validation["regex_error"],
        "n_quotes": validation["n_quotes"],
        "n_hits": validation["n_hits"],
        "hit_rate": validation["hit_rate"],
        "matching_quotes": validation["matching_quotes"],
        "accepted": accepted,
    }


# =========================================================
# DATAFRAME ORCHESTRATOR
# =========================================================

def build_regex_candidates_from_needles(
    candidate_df: pd.DataFrame,
    cfg: NeedleRegexBuilderConfig,
) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame(
            columns=[
                "needle",
                "regex_pattern",
                "seed_phrases",
                "why_this_regex",
                "confidence",
                "regex_valid",
                "regex_error",
                "n_quotes",
                "n_hits",
                "hit_rate",
                "matching_quotes",
                "accepted",
            ]
        )

    required_cols = [cfg.needle_col, cfg.quote_col]
    missing = [c for c in required_cols if c not in candidate_df.columns]
    if missing:
        raise KeyError(f"candidate_df missing required columns: {missing}")

    work = candidate_df.copy()

    if cfg.text_id_col not in work.columns:
        work[cfg.text_id_col] = [f"row_{i:04d}" for i in range(1, len(work) + 1)]

    group_cols = [cfg.needle_col]

    results: List[Dict[str, Any]] = []

    grouped = work.groupby(cfg.needle_col, dropna=False)

    for needle, g in tqdm(grouped, total=grouped.ngroups, desc="Building regex families", unit="needle"):
        needle_str = str(needle).strip()
        if not needle_str:
            continue

        quotes = (
            g[cfg.quote_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .drop_duplicates()
            .tolist()
        )

        source_texts: List[str] = []
        if cfg.text_col and cfg.text_col in g.columns:
            source_texts = (
                g[cfg.text_col]
                .fillna("")
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .drop_duplicates()
                .tolist()
            )

        res = build_regex_for_needle_group(
            needle=needle_str,
            quotes=quotes,
            source_texts=source_texts,
            cfg=cfg,
        )

        results.append(res)

    if not results:
        return pd.DataFrame(
            columns=[
                "needle",
                "regex_pattern",
                "seed_phrases",
                "why_this_regex",
                "confidence",
                "regex_valid",
                "regex_error",
                "n_quotes",
                "n_hits",
                "hit_rate",
                "matching_quotes",
                "accepted",
            ]
        )

    out_df = pd.DataFrame(results).sort_values(
        ["accepted", "hit_rate", "confidence", "needle"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    return out_df


# =========================================================
# SAVE AS JSON REGEX BANK
# =========================================================

def regex_df_to_bank(regex_df: pd.DataFrame, cfg: NeedleRegexBuilderConfig) -> Dict[str, Any]:
    bank: Dict[str, Any] = {}

    if regex_df.empty:
        return bank

    for _, r in regex_df.iterrows():
        needle = str(r.get("needle", "")).strip()
        if not needle:
            continue

        bank[needle] = {
            "pattern": str(r.get("regex_pattern", "")).strip(),
            "flags": cfg.regex_flags,
            "seed_phrases": r.get("seed_phrases", []),
            "why_this_regex": str(r.get("why_this_regex", "")).strip(),
            "confidence": float(r.get("confidence", 0.0)),
            "regex_valid": bool(r.get("regex_valid", False)),
            "n_quotes": int(r.get("n_quotes", 0)),
            "n_hits": int(r.get("n_hits", 0)),
            "hit_rate": float(r.get("hit_rate", 0.0)),
            "accepted": bool(r.get("accepted", False)),
        }

    return bank


def save_regex_bank_json(
    regex_df: pd.DataFrame,
    out_path: Path,
    cfg: NeedleRegexBuilderConfig,
) -> Path:
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bank = regex_df_to_bank(regex_df, cfg)

    out_path.write_text(
        json.dumps(bank, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_path


# =========================================================
# OPTIONAL CSV LOADER
# =========================================================

def build_regex_candidates_from_csv(
    csv_path: Path,
    cfg: NeedleRegexBuilderConfig,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return build_regex_candidates_from_needles(df, cfg)


# =========================================================
# OPTIONAL DEMO
# =========================================================

if __name__ == "__main__":
    demo_df = pd.DataFrame(
        [
            {
                "text_id": "row_0001",
                "label": "role expectation mismatch",
                "evidence_quote": "the only thing that changed between a top eval and a dismissal was my role from senior to junior",
                "text": "The only thing that changed between a top evaluation and dismissal was the role shift from senior to junior, changing expected responsibilities."
            },
            {
                "text_id": "row_0002",
                "label": "role expectation mismatch",
                "evidence_quote": "they failed to investigate if that was indeed a task senior people were supposed to perform",
                "text": "They concluded avoidance without first investigating whether ticket handling was part of the senior role."
            },
            {
                "text_id": "row_0003",
                "label": "inconsistent reasoning",
                "evidence_quote": "their explanation shifted from crashes to ticket avoidance",
                "text": "The rationale appears to move between crash counts and ticket avoidance."
            },
        ]
    )

    cfg = NeedleRegexBuilderConfig(
        model="mistral-small3.2:latest",
        needle_col="label",
        quote_col="evidence_quote",
        text_id_col="text_id",
        text_col="text",
        validation_min_hit_rate=0.50,
        validation_min_hits=1,
    )

    regex_df = build_regex_candidates_from_needles(demo_df, cfg)
    print(regex_df.to_string(index=False))