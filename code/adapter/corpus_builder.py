"""
corpus_builder.py

Stable ET corpus scanning + routing engine.
Needles + regex are meant to be injected from Jupyter during calibration.

Design goal:
- Keep calibration-varying parts out of this file (NEEDLES lists + regex text)
- Keep everything else stable and testable (scan, routing, csv writing)

Dependencies:
- pandas
- PyMuPDF (fitz)
- tqdm

Usage (Python):
    from pathlib import Path
    import re
    from corpus_builder import run_corpus_builder

    df = run_corpus_builder(
        input_root=Path("/media/hello/Vault/Tribunals/ET_Cases/"),
        matches_root=Path("/media/hello/Vault/Tribunals/_Matches"),
        needles_all=["unfair dismissal"],
        needles_any=["upheld", "verbal warning"],
        appeal_scope_regex=re.compile(r"...", re.IGNORECASE | re.VERBOSE),
        assumed_intention_regex=re.compile(r"...", re.IGNORECASE | re.VERBOSE),
    )

Usage (CLI):
    python corpus_builder.py --input-root ... --matches-root ... --needles-all "unfair dismissal" --needles-any upheld --needles-any "verbal warning" ...
    (regex via file paths; see CLI help)
"""

from __future__ import annotations

import argparse
import json
import shutil
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

    # scan knobs (can be stable defaults; optionally override from JN)
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
    regex_appeal_folder_name: str = "_APPEAL_SCOPE_REGEX"
    regex_intent_folder_name: str = "_ASSUMED_INTENTION_REGEX"
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

    # tail pages (avoid duplicates)
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
            # swallow page-level errors; keep scanning other pages
            pass

    return "\n".join(chunks)


# =========================================================
# SCANNER (worker-safe)
# =========================================================

def scan_one(
    pdf_path: str,
    *,
    needles_all: list[str],
    needles_any: list[str],
    appeal_scope_regex: Optional[Pattern[str]],
    assumed_intention_regex: Optional[Pattern[str]],
    cfg: CorpusBuilderConfig,
) -> Optional[dict[str, Any]]:
    """
    One-pass scan:
      - page count filter
      - extract head+tail text
      - require needles_all (ALL must be present)
      - compute which needles_any are present (simple substring)
      - detect appeal-scope limitation via appeal_scope_regex
      - detect assumed intention / motive via assumed_intention_regex

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

        # 1) MUST satisfy needles_all (gate)
        ok_all = all(k in text_n for k in needles_all_n) if needles_all_n else True
        if not ok_all:
            return None

        # 2) substring hits
        hit_any = [k for k in needles_any_n if k in text_n]

        # 3) regex hits (regex compiled IGNORECASE externally)
        appeal_scope_hit = False
        appeal_scope_match = ""
        if appeal_scope_regex is not None:
            m = appeal_scope_regex.search(text)
            appeal_scope_hit = bool(m)
            appeal_scope_match = (m.group(0)[:250] if m else "")

        assumed_intention_hit = False
        assumed_intention_match = ""
        if assumed_intention_regex is not None:
            mi = assumed_intention_regex.search(text)
            assumed_intention_hit = bool(mi)
            assumed_intention_match = (mi.group(0)[:250] if mi else "")

        # keep doc if it hits at least one needles_any OR either regex hit
        if (not hit_any) and (not appeal_scope_hit) and (not assumed_intention_hit):
            return None

        return {
            "ok": True,
            "error": False,

            "path": str(p),
            "pages": int(pages),
            "size_mb": round(size_mb, 3),
            "mtime": pd.to_datetime(stat.st_mtime, unit="s"),

            "hit_all": "; ".join([k for k in needles_all_n if k in text_n]),
            "hit_any": "; ".join(hit_any),

            "appeal_scope_hit": bool(appeal_scope_hit),
            "appeal_scope_match": appeal_scope_match,

            "assumed_intention_hit": bool(assumed_intention_hit),
            "assumed_intention_match": assumed_intention_match,

            # convenience
            "regex_only": bool((not hit_any) and (appeal_scope_hit or assumed_intention_hit)),
        }

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
    appeal_scope_regex: Optional[Pattern[str]],
    assumed_intention_regex: Optional[Pattern[str]],
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
                    appeal_scope_regex=appeal_scope_regex,
                    assumed_intention_regex=assumed_intention_regex,
                    cfg=cfg,
                )
            chunk = []

    for p in chunk:
        yield executor.submit(
            scan_one,
            str(p),
            needles_all=needles_all,
            needles_any=needles_any,
            appeal_scope_regex=appeal_scope_regex,
            assumed_intention_regex=assumed_intention_regex,
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
    appeal_scope_regex: Optional[Pattern[str]] = None,
    assumed_intention_regex: Optional[Pattern[str]] = None,
    cfg_overrides: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Full scan + routing + master CSV creation.

    NEEDLES + regex are meant to be passed in (from JN).
    Everything else is stable engine logic.

    Returns:
        DataFrame of matches + error rows (audit-friendly).
    """
    base_cfg = CorpusBuilderConfig(
        input_root=Path(input_root).resolve(),
        matches_root=Path(matches_root).resolve(),
    )

    if cfg_overrides:
        # dataclass is frozen -> create a new instance safely
        base_cfg = CorpusBuilderConfig(**{**base_cfg.__dict__, **cfg_overrides})

    cfg = base_cfg

    cfg.matches_root.mkdir(parents=True, exist_ok=True)

    pdfs = list(iter_pdfs(cfg.input_root))
    rows: list[dict[str, Any]] = []
    stats = Counter()

    print(f"[scan] Input root: {cfg.input_root}")
    print(f"[scan] PDFs found: {len(pdfs)}")
    print(f"[scan] NEEDLES_ALL (must match all): {needles_all}")
    print(f"[scan] NEEDLES_ANY (substring OR regex): {needles_any}")
    print(f"[scan] Pages >= {cfg.min_pages}")
    print(f"[scan] Text scan: head={cfg.text_pages_head} pages, tail={cfg.text_pages_tail} pages")
    print(f"[scan] Workers={cfg.max_workers} | submit_chunk={cfg.submit_chunk_size}")
    print(f"[out] Principal matches folder: {cfg.matches_root}")
    print(f"[out] Preserve structure: {cfg.preserve_structure}")

    # 1) Scan once in parallel (chunked submission)
    with ProcessPoolExecutor(max_workers=cfg.max_workers) as ex:
        futs = list(
            _submit_in_chunks(
                ex,
                pdfs,
                chunk_size=cfg.submit_chunk_size,
                needles_all=needles_all,
                needles_any=needles_any,
                appeal_scope_regex=appeal_scope_regex,
                assumed_intention_regex=assumed_intention_regex,
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
                rows.append(r)  # keep error rows for audit
                continue

            stats["match"] += 1
            if r.get("regex_only"):
                stats["match_regex_only"] += 1
            if r.get("appeal_scope_hit"):
                stats["match_appeal_scope_regex"] += 1
            if r.get("assumed_intention_hit"):
                stats["match_assumed_intention_regex"] += 1
            if (r.get("hit_any") or "").strip():
                stats["match_substring_any"] += 1

            rows.append(r)

    df = pd.DataFrame(rows)
    if df.empty:
        print("[scan] Matches: 0 (and no error rows)")
        return df

    # Split out errors so they don't pollute the match routing
    df_err = df[df.get("error", False) == True].copy() if "error" in df.columns else df.iloc[0:0].copy()
    df_ok = df[df.get("ok", False) == True].copy() if "ok" in df.columns else df.iloc[0:0].copy()

    print(
        f"[stats] scanned={len(pdfs)} | match={stats['match']} | no_match={stats['no_match']} | error={stats['error']}"
    )
    print(
        f"[stats] substring_any={stats['match_substring_any']} | appeal_scope_regex={stats['match_appeal_scope_regex']} | "
        f"assumed_intention_regex={stats['match_assumed_intention_regex']} | regex_only={stats['match_regex_only']}"
    )

    if df_ok.empty:
        print("[scan] No OK matches (only errors). Writing CSV only.")
    else:
        df_ok = df_ok.sort_values(["pages", "size_mb"], ascending=False).reset_index(drop=True)
        print(f"[scan] OK matches: {len(df_ok)}")

    # 2) Create one subfolder per NEEDLES_ANY term and copy files into each
    needles_any_norm = [_norm(x, cfg.case_sensitive) for x in needles_any if x and x.strip()]
    norm_to_original = {_norm(x, cfg.case_sensitive): x for x in needles_any if x and x.strip()}

    if not df_ok.empty:
        # For routing, explode hit_any into a list (may be empty if only regex hit)
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

        # 2b) Copy ALL appeal-scope regex hits into ONE dedicated folder
        if "appeal_scope_hit" in df_ok.columns:
            df_regex = df_ok[df_ok["appeal_scope_hit"] == True].copy()
        else:
            df_regex = df_ok.iloc[0:0].copy()

        print(f"[regex] APPEAL_SCOPE_REGEX hits: {len(df_regex)}")

        if not df_regex.empty:
            out_dir = (cfg.matches_root / cfg.regex_appeal_folder_name).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            for src_str in tqdm(df_regex["path"].tolist(), desc=f"Copying -> {cfg.regex_appeal_folder_name}", unit="file"):
                src = Path(src_str)
                copy_to_subfolder(
                    src=src,
                    in_root=cfg.input_root,
                    subfolder=out_dir,
                    preserve_structure=cfg.preserve_structure,
                )

        # 2c) Copy ALL assumed-intention regex hits into ONE dedicated folder
        if "assumed_intention_hit" in df_ok.columns:
            df_intent = df_ok[df_ok["assumed_intention_hit"] == True].copy()
        else:
            df_intent = df_ok.iloc[0:0].copy()

        print(f"[regex] ASSUMED_INTENTION_REGEX hits: {len(df_intent)}")

        if not df_intent.empty:
            out_dir = (cfg.matches_root / cfg.regex_intent_folder_name).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)

            for src_str in tqdm(df_intent["path"].tolist(), desc=f"Copying -> {cfg.regex_intent_folder_name}", unit="file"):
                src = Path(src_str)
                copy_to_subfolder(
                    src=src,
                    in_root=cfg.input_root,
                    subfolder=out_dir,
                    preserve_structure=cfg.preserve_structure,
                )

        # 2d) Copy regex-only hits into ONE dedicated folder
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

    # 3) Write ONE master CSV in MATCHES_ROOT (and nowhere else)
    df_out = df.copy()

    if "hit_any_list" in df_out.columns:
        df_out = df_out.drop(columns=["hit_any_list"], errors="ignore")

    df_out["matches_root"] = str(cfg.matches_root)

    # boolean columns per NEEDLES_ANY (from hit_any)
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

    # boolean column for appeal-scope regex
    df_out["has__appeal_scope_regex"] = (
    df_out["appeal_scope_hit"].fillna(False).astype(bool)
    if "appeal_scope_hit" in df_out.columns
    else False
)

    df_out["has__assumed_intention_regex"] = (
        df_out["assumed_intention_hit"].fillna(False).astype(bool)
        if "assumed_intention_hit" in df_out.columns
        else False
    )
    # ANY flag (what Moltie would use)
    # NOTE: must build the column list based on the original needles_any labels
    needle_cols = [f"has__{_slugify(x)}" for x in needles_any]

    extra_cols = []
    if "has__appeal_scope_regex" in df_out.columns:
        extra_cols.append("has__appeal_scope_regex")
    if "has__assumed_intention_regex" in df_out.columns:
        extra_cols.append("has__assumed_intention_regex")

    cols_to_check = [c for c in needle_cols if c in df_out.columns] + extra_cols

    df_out["has__any_needle"] = (
        df_out[cols_to_check]
        .fillna(False)
        .any(axis=1)
    )

    master_path = cfg.matches_root / cfg.master_csv_name
    df_out.to_csv(master_path, index=False, quoting=1)  # csv.QUOTE_ALL = 1
    print(f"[csv] Wrote single master CSV: {master_path}")

    # 4) Optional: print a quick error preview
    if not df_err.empty:
        print("\n[warn] Some PDFs failed parsing/opening. First 10 errors:")
        cols = [c for c in ["path", "error_msg"] if c in df_err.columns]
        print(df_err[cols].head(10).to_string(index=False))

    return df_out


# =========================================================
# CLI (optional, but useful)
# =========================================================

def _load_regex_from_file(path: Optional[str]) -> Optional[Pattern[str]]:
    """
    Loads a regex pattern from a text file.
    File can be raw regex, or JSON {"pattern": "...", "flags": ["IGNORECASE","VERBOSE"]}.
    """
    if not path:
        return None

    p = Path(path).expanduser().resolve()
    raw = p.read_text(encoding="utf-8").strip()

    # JSON mode
    if raw.startswith("{"):
        obj = json.loads(raw)
        pattern = obj["pattern"]
        flags_list = obj.get("flags", [])
        flags = 0
        for f in flags_list:
            f = f.upper().strip()
            if f == "IGNORECASE":
                import re
                flags |= re.IGNORECASE
            elif f == "VERBOSE":
                import re
                flags |= re.VERBOSE
            elif f == "MULTILINE":
                import re
                flags |= re.MULTILINE
            elif f == "DOTALL":
                import re
                flags |= re.DOTALL
        import re
        return re.compile(pattern, flags)

    # raw regex mode (default ignorecase+verbose is NOT assumed)
    import re
    return re.compile(raw)


def main():
    ap = argparse.ArgumentParser(description="ET PDF corpus builder (needle + regex scanner).")
    ap.add_argument("--input-root", required=True, type=str)
    ap.add_argument("--matches-root", required=True, type=str)

    ap.add_argument("--needles-all", action="append", default=[], help="Gate needles: ALL must appear (repeatable).")
    ap.add_argument("--needles-any", action="append", default=[], help="Bucket needles: ANY may appear (repeatable).")

    ap.add_argument("--appeal-scope-regex-file", type=str, default=None, help="Path to regex file (raw or JSON).")
    ap.add_argument("--assumed-intention-regex-file", type=str, default=None, help="Path to regex file (raw or JSON).")

    # optional overrides
    ap.add_argument("--min-pages", type=int, default=None)
    ap.add_argument("--head-pages", type=int, default=None)
    ap.add_argument("--tail-pages", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--submit-chunk", type=int, default=None)
    ap.add_argument("--preserve-structure", type=int, default=None, help="1 preserve structure, 0 flatten.")

    args = ap.parse_args()

    appeal_re = _load_regex_from_file(args.appeal_scope_regex_file)
    intent_re = _load_regex_from_file(args.assumed_intention_regex_file)

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
        appeal_scope_regex=appeal_re,
        assumed_intention_regex=intent_re,
        cfg_overrides=overrides or None,
    )

    if isinstance(df, pd.DataFrame) and not df.empty:
        print(df.head(25).to_string(index=False))


if __name__ == "__main__":
    main()