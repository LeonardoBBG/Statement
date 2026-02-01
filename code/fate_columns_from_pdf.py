from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple


def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        import pdfplumber  # type: ignore
        chunks: List[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                chunks.append(page.extract_text() or "")
        text = "\n".join(chunks).strip()
        if not text:
            raise ValueError("Empty extracted text.")
        return text
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(str(pdf_path))
        chunks = []
        for page in reader.pages:
            chunks.append(page.extract_text() or "")
        text = "\n".join(chunks).strip()
        if not text:
            raise ValueError("Empty extracted text.")
        return text
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract text from PDF. If scanned-only, you need a text-based export. Error: {e}"
        ) from e


def _norm(s: str) -> str:
    s = s.replace("\xa0", " ").replace("\u2014", "—").replace("\u2013", "—")
    return " ".join(s.split()).strip()


def _split_into_parenthesized_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Returns list of (marker, block_text) where marker is the inside of leading (...) and
    block_text includes the marker + following text until next marker.
    We scan for occurrences of "(...)" and slice between them.
    """
    raw = text.replace("\u2014", "—").replace("\u2013", "—")
    raw = raw.replace("\xa0", " ")

    # Find all markers like "(...)" at ANY position; we then slice between their spans.
    marker_pat = re.compile(r"\(([^)]+)\)")
    matches = list(marker_pat.finditer(raw))

    blocks: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        marker = _norm(m.group(1))
        block = _norm(raw[start:end])
        blocks.append((marker, block))

    return blocks


def parse_fate_columns_fulltext(text: str) -> Dict[str, Any]:
    """
    Extract columns as FULL TEXT blocks:
    - clusters: blocks whose marker is NOT KEG_XX — ...
    - kegs: blocks whose marker begins with KEG_XX — ...
    """
    blocks = _split_into_parenthesized_blocks(text)

    # KEG marker example: "KEG_01 — Claimant's appeal against dismissal"
    keg_marker_pat = re.compile(r"^(KEG_\d{2})\s*—\s*(.+)$")

    clusters: List[Dict[str, str]] = []
    kegs: List[Dict[str, str]] = []

    seen_cluster_headings = set()
    seen_keg_ids = set()

    for marker, block in blocks:
        mm = keg_marker_pat.match(marker)
        if mm:
            keg_id = _norm(mm.group(1))
            keg_title = _norm(mm.group(2))
            if keg_id in seen_keg_ids:
                continue
            kegs.append({
                "id": keg_id,
                "title": keg_title,
                "full_text": block
            })
            seen_keg_ids.add(keg_id)
        else:
            # Filter out obvious non-columns (e.g., "(Case Analysis)" might be a heading you *do* want or not)
            # We only keep blocks that look like the FATE issue statements:
            # In your file they are like "(Intentional Crashes) is ..."
            # We'll keep any non-KEG marker, but dedupe by marker to avoid repeats.
            heading = marker
            if heading in seen_cluster_headings:
                continue
            clusters.append({
                "id": heading,          # stable id = heading text (can be replaced with CL_01 later)
                "heading": heading,
                "full_text": block
            })
            seen_cluster_headings.add(heading)

    if not clusters and not kegs:
        raise ValueError("No cluster/KEG blocks extracted. PDF format may differ or extraction failed.")

    payload = {
        "meta": {
            "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_clusters": len(clusters),
            "n_kegs": len(kegs),
        },
        "clusters": clusters,
        "kegs": kegs,
    }
    return payload


def write_outputs(payload: Dict[str, Any], out_dir: Path, source_pdf: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload["meta"]["source_pdf"] = str(source_pdf)

    json_path = out_dir / "fate_columns.json"
    csv_path = out_dir / "fate_columns.csv"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV mirror (kept simple but includes full_text)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "id", "title_or_heading", "full_text"])
        for c in payload["clusters"]:
            w.writerow(["cluster", c["id"], c["heading"], c["full_text"]])
        for k in payload["kegs"]:
            w.writerow(["keg", k["id"], k["title"], k["full_text"]])

    return json_path, csv_path


def main() -> int:
    p = argparse.ArgumentParser(description="Extract FATE columns (clusters + KEGs) with FULL TEXT from a PDF.")
    p.add_argument("--pdf", required=True, help="Path to FATE output PDF")
    p.add_argument("--out_dir", required=True, help="Directory to write fate_columns.json/csv")
    args = p.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text = extract_text_from_pdf(pdf_path)
    payload = parse_fate_columns_fulltext(text)
    json_path, csv_path = write_outputs(payload, out_dir, pdf_path)

    print(f"Clusters: {payload['meta']['n_clusters']}")
    print(f"KEGs:     {payload['meta']['n_kegs']}")
    print(f"Wrote:\n  {json_path}\n  {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
