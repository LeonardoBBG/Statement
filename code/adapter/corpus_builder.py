"""
corpus_builder.py

Stable ET corpus scanning + routing engine.
Needles + regex are meant to be injected from Jupyter during calibration.

Design rule:
- needles_all   -> list[str]
- needles_any   -> list[str]
- regex_buckets -> dict[str, Pattern[str]]

Design goal:
- Keep calibration-varying parts out of this file
  (needle lists + regex text/patterns)
- Keep everything else stable and testable
  (scan, routing, csv writing)

Dependencies:
- pandas
- PyMuPDF (fitz)
- tqdm
"""

from __future__ import annotations

import argparse
import json
import shutil
import re
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from typing import Iterable, Optional, Pattern, Any

import pandas as pd
import fitz  # PyMuPDF
from tqdm import tqdm


# =========================================================
# CONFIG STRUCT
# =========================================================

@dataclass(frozen=True)
class CorpusBuilderConfig:
    input_root: Path
    matches_root: Path

    # scan knobs
    case_sensitive: bool = False
    min_pages: int = 4
    text_pages_head: int = 12
    text_pages_tail: int = 6

    # parallel knobs
    max_workers: int = 24
    submit_chunk_size: int = 2000

    # output knobs
    preserve_structure: bool = True
    master_csv_name: str = "_matches_index.csv"

    # special folders
    regex_only_folder_name: str = "_REGEX_ONLY"


# =========================================================
# UTILS
# =========================================================

def _norm(s: str, case_sensitive: bool) -> str:
    return s if case_sensitive else s.lower()


def _slugify(s: str) -> str:
    s = (s or "").strip().replace(" ", "_")
    s = "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-", "."))
    return s[:120] if s else "EMPTY"


def iter_pdfs(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.pdf"):
        if p.is_file():
            yield p


def _safe_copy(src: Path, dst: Path) -> Path:
    """
    Copy src to dst. If dst exists, add suffix _1, _2, ...
    Returns final path.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    stem, suffix = dst.stem, dst.suffix
    i = 1
    while True:
        cand = dst.with_name(f"{stem}_{i}{suffix}")
        if not cand.exists():
            shutil.copy2(src, cand)
            return cand
        i += 1


def copy_to_subfolder(src: Path, in_root: Path, subfolder: Path, preserve_structure: bool = True) -> Path:
    """
    Copy one PDF into subfolder, optionally preserving structure.
    Returns final copied path.
    """
    if preserve_structure:
        try:
            rel = src.relative_to(in_root)
        except ValueError:
            rel = Path(src.name)
        dst = subfolder / rel
    else:
        dst = subfolder / src.name

    return _safe_copy(src, dst)


def _extract_text_head_tail(doc: fitz.Document, head: int, tail: int) -> str:
    pages = doc.page_count
    head_n = min(max(head, 0), pages)
    tail_n = min(max(tail, 0), pages)

    idxs = list(range(head_n))

    if tail_n > 0:
        start = max(pages - tail_n, 0)
        tail_idxs = list(range(start, pages))
        for i in tail_idxs:
            if i not in idxs:
                idxs.append(i)

    chunks: list[str] = []
    for i in idxs:
        try:
            chunks.append(doc.load_page(i).get_text("text"))
        except Exception:
            pass

    return "\n".join(chunks)


def _normalize_regex_folder_map(regex_buckets: dict[str, Pattern[str]],
                                regex_folder_map: Optional[dict[str, str]]) -> dict[str, str]:
    """
    Ensure every regex bucket has a folder name.
    """
    out: dict[str, str] = {}
    regex_folder_map = regex_folder_map or {}
    for bucket_name in regex_buckets.keys():
        out[bucket_name] = regex_folder_map.get(bucket_name, f"_{_slugify(bucket_name).upper()}")
    return out


# =========================================================
# SCANNER (worker-safe)
# =========================================================

def scan_one(
    pdf_path: str,
    *,
    needles_all: list[str],
    needles_any: list[str],
    regex_buckets: Optional[dict[str, Pattern[str]]],
    cfg: CorpusBuilderConfig,
) -> Optional[dict[str, Any]]:
    """
    One-pass scan:
      - page count filter
      - extract head+tail text
      - require needles_all (ALL must be present)
      - compute which needles_any are present (simple substring)
      - detect regex bucket hits dynamically

    Returns:
      - if matched: { ... matched fields ..., ok=True, error=False }
      - if not matched: None
      - if error: { ok=False, error=True, error_msg=..., path=... }
    """
    p = Path(pdf_path)

    try:
        stat = p.stat()
        size_mb = stat.st_size / (1024 * 1024)

        doc = fitz.open(p)
        pages = doc.page_count
        if pages < cfg.min_pages:
            doc.close()
            return None

        text = _extract_text_head_tail(doc, cfg.text_pages_head, cfg.text_pages_tail)
        doc.close()

        text_n = _norm(text, cfg.case_sensitive)

        needles_all_n = [_norm(x, cfg.case_sensitive) for x in needles_all if x and x.strip()]
        needles_any_n = [_norm(x, cfg.case_sensitive) for x in needles_any if x and x.strip()]

        # 1) MUST satisfy needles_all
        ok_all = all(k in text_n for k in needles_all_n) if needles_all_n else True
        if not ok_all:
            return None

        # 2) substring hits
        hit_any = [k for k in needles_any_n if k in text_n]

        # 3) dynamic regex hits
        regex_buckets = regex_buckets or {}
        regex_hits: dict[str, bool] = {}
        regex_matches: dict[str, str] = {}

        any_regex_hit = False
        for bucket_name, bucket_re in regex_buckets.items():
            m = bucket_re.search(text) if bucket_re is not None else None
            hit = bool(m)
            regex_hits[bucket_name] = hit
            regex_matches[bucket_name] = m.group(0)[:250] if m else ""
            if hit:
                any_regex_hit = True

        # keep doc if it hits at least one needles_any OR at least one regex bucket
        if (not hit_any) and (not any_regex_hit):
            return None

        row: dict[str, Any] = {
            "ok": True,
            "error": False,

            "path": str(p),
            "pages": int(pages),
            "size_mb": round(size_mb, 3),
            "mtime": pd.to_datetime(stat.st_mtime, unit="s"),

            "hit_all": "; ".join([k for k in needles_all_n if k in text_n]),
            "hit_any": "; ".join(hit_any),

            "regex_only": bool((not hit_any) and any_regex_hit),
        }

        for bucket_name in regex_buckets.keys():
            row[f"regex_hit__{_slugify(bucket_name)}"] = bool(regex_hits.get(bucket_name, False))
            row[f"regex_match__{_slugify(bucket_name)}"] = regex_matches.get(bucket_name, "")

        return row

    except Exception as e:
        return {
            "ok": False,
            "error": True,
            "error_msg": f"{type(e).__name__}: {str(e)[:300]}",
            "path": str(p),
        }


def _submit_in_chunks(
    executor: ProcessPoolExecutor,
    items: Iterable[Path],
    *,
    chunk_size: int,
    needles_all: list[str],
    needles_any: list[str],
    regex_buckets: Optional[dict[str, Pattern[str]]],
    cfg: CorpusBuilderConfig,
):
    """
    Yield futures, but avoid submitting everything at once (RAM-friendly).
    """
    chunk: list[Path] = []

    for it in items:
        chunk.append(it)
        if len(chunk) >= chunk_size:
            for p in chunk:
                yield executor.submit(
                    scan_one,
                    str(p),
                    needles_all=needles_all,
                    needles_any=needles_any,
                    regex_buckets=regex_buckets,
                    cfg=cfg,
                )
            chunk = []

    for p in chunk:
        yield executor.submit(
            scan_one,
            str(p),
            needles_all=needles_all,
            needles_any=needles_any,
            regex_buckets=regex_buckets,
            cfg=cfg,
        )


# =========================================================
# CORE RUNNER
# =========================================================

def run_corpus_builder(
    *,
    input_root: Path,
    matches_root: Path,
    needles_all: list[str],
    needles_any: list[str],
    regex_buckets: Optional[dict[str, Pattern[str]]] = None,
    regex_folder_map: Optional[dict[str, str]] = None,
    cfg_overrides: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Full scan + routing + master CSV creation.

    Design rule:
      - needles_all   -> list[str]
      - needles_any   -> list[str]
      - regex_buckets -> dict[str, Pattern[str]]

    Returns:
        DataFrame of matches + error rows (audit-friendly).
    """
    base_cfg = CorpusBuilderConfig(
        input_root=Path(input_root).resolve(),
        matches_root=Path(matches_root).resolve(),
    )

    if cfg_overrides:
        base_cfg = CorpusBuilderConfig(**{**base_cfg.__dict__, **cfg_overrides})

    cfg = base_cfg
    cfg.matches_root.mkdir(parents=True, exist_ok=True)

    regex_buckets = regex_buckets or {}
    regex_folder_map = _normalize_regex_folder_map(regex_buckets, regex_folder_map)

    pdfs = list(iter_pdfs(cfg.input_root))
    rows: list[dict[str, Any]] = []
    stats = Counter()

    print(f"[scan] Input root: {cfg.input_root}")
    print(f"[scan] PDFs found: {len(pdfs)}")
    print(f"[scan] NEEDLES_ALL (must match all): {needles_all}")
    print(f"[scan] NEEDLES_ANY (substring OR regex): {needles_any}")
    print(f"[scan] REGEX_BUCKETS: {list(regex_buckets.keys())}")
    print(f"[scan] Pages >= {cfg.min_pages}")
    print(f"[scan] Text scan: head={cfg.text_pages_head} pages, tail={cfg.text_pages_tail} pages")
    print(f"[scan] Workers={cfg.max_workers} | submit_chunk={cfg.submit_chunk_size}")
    print(f"[out] Principal matches folder: {cfg.matches_root}")
    print(f"[out] Preserve structure: {cfg.preserve_structure}")

    # 1) Scan once in parallel
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
        futs = list(
            _submit_in_chunks(
                ex,
                pdfs,
                chunk_size=cfg.submit_chunk_size,
                needles_all=needles_all,
                needles_any=needles_any,
                regex_buckets=regex_buckets,
                cfg=cfg,
            )
        )

        for fut in tqdm(as_completed(futs), total=len(futs), desc="Scanning PDFs", unit="file"):
            r = fut.result()
            if r is None:
                stats["no_match"] += 1
                continue

            if r.get("error"):
                stats["error"] += 1
                rows.append(r)
                continue

            stats["match"] += 1
            if r.get("regex_only"):
                stats["match_regex_only"] += 1
            if (r.get("hit_any") or "").strip():
                stats["match_substring_any"] += 1

            for bucket_name in regex_buckets.keys():
                hit_col = f"regex_hit__{_slugify(bucket_name)}"
                if r.get(hit_col):
                    stats[f"match_regex__{bucket_name}"] += 1

            rows.append(r)

    df = pd.DataFrame(rows)
    if df.empty:
        print("[scan] Matches: 0 (and no error rows)")
        return df

    df_err = df[df.get("error", False) == True].copy() if "error" in df.columns else df.iloc[0:0].copy()
    df_ok = df[df.get("ok", False) == True].copy() if "ok" in df.columns else df.iloc[0:0].copy()

    print(
        f"[stats] scanned={len(pdfs)} | match={stats['match']} | no_match={stats['no_match']} | error={stats['error']}"
    )
    print(f"[stats] substring_any={stats['match_substring_any']} | regex_only={stats['match_regex_only']}")
    for bucket_name in regex_buckets.keys():
        print(f"[stats] regex::{bucket_name}={stats[f'match_regex__{bucket_name}']}")

    if df_ok.empty:
        print("[scan] No OK matches (only errors). Writing CSV only.")
    else:
        df_ok = df_ok.sort_values(["pages", "size_mb"], ascending=False).reset_index(drop=True)
        print(f"[scan] OK matches: {len(df_ok)}")

    # 2) Create one subfolder per NEEDLES_ANY term
    needles_any_norm = [_norm(x, cfg.case_sensitive) for x in needles_any if x and x.strip()]
    norm_to_original = {_norm(x, cfg.case_sensitive): x for x in needles_any if x and x.strip()}

    if not df_ok.empty:
        df_ok["hit_any_list"] = df_ok.get("hit_any", "").fillna("").apply(
            lambda s: [x.strip() for x in s.split(";") if x.strip()]
        )

        for needle_norm in needles_any_norm:
            label = norm_to_original[needle_norm]
            folder_name = _slugify(label)
            out_dir = (cfg.matches_root / folder_name).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            mask = df_ok["hit_any_list"].apply(lambda xs: needle_norm in xs)
            df_group = df_ok[mask]

            if df_group.empty:
                print(f"[group] '{label}' -> 0 matches (skip)")
                continue

            print(f"[group] '{label}' -> {len(df_group)} matches -> {out_dir}")

            for src_str in tqdm(df_group["path"].tolist(), desc=f"Copying -> {folder_name}", unit="file"):
                src = Path(src_str)
                copy_to_subfolder(
                    src=src,
                    in_root=cfg.input_root,
                    subfolder=out_dir,
                    preserve_structure=cfg.preserve_structure,
                )

        # 2b) Dynamic regex folders
        for bucket_name in regex_buckets.keys():
            hit_col = f"regex_hit__{_slugify(bucket_name)}"
            folder_name = regex_folder_map[bucket_name]

            if hit_col in df_ok.columns:
                df_regex = df_ok[df_ok[hit_col] == True].copy()
            else:
                df_regex = df_ok.iloc[0:0].copy()

            print(f"[regex] {bucket_name} hits: {len(df_regex)}")

            if not df_regex.empty:
                out_dir = (cfg.matches_root / folder_name).resolve()
                out_dir.mkdir(parents=True, exist_ok=True)

                for src_str in tqdm(df_regex["path"].tolist(), desc=f"Copying -> {folder_name}", unit="file"):
                    src = Path(src_str)
                    copy_to_subfolder(
                        src=src,
                        in_root=cfg.input_root,
                        subfolder=out_dir,
                        preserve_structure=cfg.preserve_structure,
                    )

        # 2c) Regex-only folder
        if "regex_only" in df_ok.columns:
            df_regex_only = df_ok[df_ok["regex_only"] == True].copy()
        else:
            df_regex_only = df_ok.iloc[0:0].copy()

        print(f"[regex] REGEX_ONLY (no substring NEEDLES_ANY) hits: {len(df_regex_only)}")

        if not df_regex_only.empty:
            out_dir = (cfg.matches_root / cfg.regex_only_folder_name).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            for src_str in tqdm(df_regex_only["path"].tolist(), desc=f"Copying -> {cfg.regex_only_folder_name}", unit="file"):
                src = Path(src_str)
                copy_to_subfolder(
                    src=src,
                    in_root=cfg.input_root,
                    subfolder=out_dir,
                    preserve_structure=cfg.preserve_structure,
                )

    # 3) Write master CSV
    df_out = df.copy()

    if "hit_any_list" in df_out.columns:
        df_out = df_out.drop(columns=["hit_any_list"], errors="ignore")

    df_out["matches_root"] = str(cfg.matches_root)

    # boolean columns per NEEDLES_ANY
    needles_any_norm = [_norm(x, cfg.case_sensitive) for x in needles_any if x and x.strip()]
    norm_to_original = {_norm(x, cfg.case_sensitive): x for x in needles_any if x and x.strip()}

    for needle_norm in needles_any_norm:
        label = norm_to_original[needle_norm]
        col = f"has__{_slugify(label)}"
        hit_series = df_out["hit_any"] if "hit_any" in df_out.columns else ""

        if isinstance(hit_series, str):
            df_out[col] = False
        else:
            df_out[col] = hit_series.fillna("").apply(
                lambda s: needle_norm in [x.strip() for x in s.split(";") if x.strip()]
            )

    # boolean columns per regex bucket
    regex_flag_cols: list[str] = []
    for bucket_name in regex_buckets.keys():
        raw_hit_col = f"regex_hit__{_slugify(bucket_name)}"
        final_col = f"has__{_slugify(bucket_name)}"
        regex_flag_cols.append(final_col)

        if raw_hit_col in df_out.columns:
            df_out[final_col] = df_out[raw_hit_col].fillna(False).astype(bool)
        else:
            df_out[final_col] = False

    # ANY flag
    needle_cols = [f"has__{_slugify(x)}" for x in needles_any]
    cols_to_check = [c for c in needle_cols if c in df_out.columns] + [c for c in regex_flag_cols if c in df_out.columns]

    if cols_to_check:
        df_out["has__any_needle"] = df_out[cols_to_check].fillna(False).any(axis=1)
    else:
        df_out["has__any_needle"] = False

    master_path = cfg.matches_root / cfg.master_csv_name
    df_out.to_csv(master_path, index=False, quoting=1)  # csv.QUOTE_ALL = 1
    print(f"[csv] Wrote single master CSV: {master_path}")

    if not df_err.empty:
        print("\n[warn] Some PDFs failed parsing/opening. First 10 errors:")
        cols = [c for c in ["path", "error_msg"] if c in df_err.columns]
        print(df_err[cols].head(10).to_string(index=False))

    return df_out


# =========================================================
# CLI HELPERS
# =========================================================

def _compile_regex_from_spec(spec: dict[str, Any]) -> Pattern[str]:
    flags = 0
    for f in spec.get("flags", []):
        fu = str(f).upper().strip()
        if fu == "IGNORECASE":
            flags |= re.IGNORECASE
        elif fu == "VERBOSE":
            flags |= re.VERBOSE
        elif fu == "MULTILINE":
            flags |= re.MULTILINE
        elif fu == "DOTALL":
            flags |= re.DOTALL

    return re.compile(spec["pattern"], flags)


def _load_regex_buckets_from_json(path: Optional[str]) -> dict[str, Pattern[str]]:
    """
    Load regex buckets from JSON file shaped like:
    {
      "bucket_name": {"pattern": "...", "flags": ["IGNORECASE", "VERBOSE"]},
      ...
    }
    """
    if not path:
        return {}

    p = Path(path).expanduser().resolve()
    obj = json.loads(p.read_text(encoding="utf-8"))

    out: dict[str, Pattern[str]] = {}
    for bucket_name, spec in obj.items():
        out[bucket_name] = _compile_regex_from_spec(spec)
    return out


def main():
    ap = argparse.ArgumentParser(description="ET PDF corpus builder (needle + regex scanner).")
    ap.add_argument("--input-root", required=True, type=str)
    ap.add_argument("--matches-root", required=True, type=str)

    ap.add_argument("--needles-all", action="append", default=[], help="Gate needles: ALL must appear (repeatable).")
    ap.add_argument("--needles-any", action="append", default=[], help="Bucket needles: ANY may appear (repeatable).")
    ap.add_argument("--regex-buckets-json", type=str, default=None, help="Path to JSON file containing regex bucket specs.")

    # optional overrides
    ap.add_argument("--min-pages", type=int, default=None)
    ap.add_argument("--head-pages", type=int, default=None)
    ap.add_argument("--tail-pages", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--submit-chunk", type=int, default=None)
    ap.add_argument("--preserve-structure", type=int, default=None, help="1 preserve structure, 0 flatten.")

    args = ap.parse_args()

    regex_buckets = _load_regex_buckets_from_json(args.regex_buckets_json)

    overrides: dict[str, Any] = {}
    if args.min_pages is not None:
        overrides["min_pages"] = int(args.min_pages)
    if args.head_pages is not None:
        overrides["text_pages_head"] = int(args.head_pages)
    if args.tail_pages is not None:
        overrides["text_pages_tail"] = int(args.tail_pages)
    if args.workers is not None:
        overrides["max_workers"] = int(args.workers)
    if args.submit_chunk is not None:
        overrides["submit_chunk_size"] = int(args.submit_chunk)
    if args.preserve_structure is not None:
        overrides["preserve_structure"] = bool(int(args.preserve_structure))

    df = run_corpus_builder(
        input_root=Path(args.input_root),
        matches_root=Path(args.matches_root),
        needles_all=args.needles_all,
        needles_any=args.needles_any,
        regex_buckets=regex_buckets,
        regex_folder_map=None,
        cfg_overrides=overrides or None,
    )

    if isinstance(df, pd.DataFrame) and not df.empty:
        print(df.head(25).to_string(index=False))


if __name__ == "__main__":
    main()