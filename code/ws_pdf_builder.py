# ws_pdf_builder.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase.pdfmetrics import stringWidth


# =============================================================================
# JSON loader (.jsonl / .json)
# =============================================================================

def load_items(json_path: str) -> List[Dict[str, Any]]:
    """
    Loads .jsonl or .json.

    Expected schema (per row):
      {
        "section_title": str,
        "section_summary": str,
        "paragraphs": [
            {"p": int, "text_verbatim": str, ...},
            ...
        ],
        "_source_row": int
      }
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"JSON_PATH not found: {json_path}")

    suf = p.suffix.lower()

    if suf == ".jsonl":
        out: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    if suf == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list):
            return obj["items"]
        if isinstance(obj, dict):
            # fallback
            return [obj]
        raise ValueError("Unsupported JSON structure")

    raise ValueError("JSON_PATH must end with .jsonl or .json")


# =============================================================================
# Text drawing helpers (incl. justification)
# =============================================================================

def _wrap_lines(text: str, font: str, size: int, max_w: float) -> List[str]:
    return simpleSplit(text, font, size, max_w)


def _draw_justified_paragraph(
    c: canvas.Canvas,
    lines: List[str],
    x: float,
    y: float,
    max_w: float,
    font: str,
    size: int,
    leading: int,
) -> float:
    """
    Full justification for all lines except the last line of the paragraph.
    """
    c.setFont(font, size)

    for i, line in enumerate(lines):
        if not line.strip():
            y -= leading
            continue

        last_line = (i == len(lines) - 1)
        spaces = line.count(" ")

        # last line OR no spaces -> normal
        if last_line or spaces == 0:
            c.drawString(x, y, line)
            y -= leading
            continue

        w = stringWidth(line, font, size)
        extra = max_w - w

        # If overflow or exact, draw normally
        if extra <= 0:
            c.drawString(x, y, line)
            y -= leading
            continue

        # Spread extra across spaces
        word_space = extra / spaces
        t = c.beginText(x, y)
        t.setFont(font, size)
        t.setWordSpace(word_space)
        t.textOut(line)
        c.drawText(t)
        y -= leading

    return y


def draw_paragraph(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_w: float,
    font: str,
    size: int,
    leading: int,
    justify: bool,
) -> float:
    """
    Wraps and draws a paragraph. Returns new y.
    """
    # Preserve existing newlines intentionally:
    # split into "blocks" and draw each block as its own paragraph.
    blocks = text.split("\n")
    for b in blocks:
        b = b.rstrip()
        if b == "":
            y -= leading  # blank line
            continue

        lines = _wrap_lines(b, font, size, max_w)
        if justify:
            y = _draw_justified_paragraph(c, lines, x, y, max_w, font, size, leading)
        else:
            c.setFont(font, size)
            for line in lines:
                c.drawString(x, y, line)
                y -= leading

    return y


# =============================================================================
# Main PDF generator
# =============================================================================

def generate_ws_pdf(
    json_path: str,
    out_pdf: str,
    analysis: bool,
    *,
    # PDF layout
    page_size: Tuple[float, float] = A4,
    left: float = 2.0 * cm,
    right: float = 2.0 * cm,
    top: float = 2.0 * cm,
    bottom: float = 2.0 * cm,
    font: str = "Times-Roman",
    font_bold: str = "Times-Bold",
    size: int = 11,
    leading: int = 14,
    # New feature
    justify_body: bool = True,
    # Controls
    render_section_summary: bool = True,
) -> None:
    """
    Renders your WS JSONL/JSON into a PDF.

    analysis=True  -> analysis pack (no numbering)
    analysis=False -> clean ET WS (numbers each paragraph)

    justify_body=True -> full justification (except last line per paragraph)
    """
    items = load_items(json_path)

    page_w, page_h = page_size
    max_w = page_w - left - right

    c = canvas.Canvas(out_pdf, pagesize=page_size)
    y = page_h - top

    def new_page() -> float:
        c.showPage()
        c.setFont(font, size)
        return page_h - top

    c.setFont(font, size)

    para_counter = 1

    for sec in items:
        # --- Section title (bold) ---
        title = (sec.get("section_title") or "").strip()
        if title:
            if y < bottom + (leading * 2):
                y = new_page()
            c.setFont(font_bold, size)
            y = draw_paragraph(
                c, title, left, y, max_w,
                font_bold, size, leading,
                justify=False
            )
            c.setFont(font, size)
            y -= int(0.4 * leading)

        # --- Optional section summary (normal) ---
        if render_section_summary:
            summary = (sec.get("section_summary") or "").strip()
            if summary:
                if y < bottom + (leading * 3):
                    y = new_page()
                y = draw_paragraph(
                    c, summary, left, y, max_w,
                    font, size, leading,
                    justify=bool(justify_body)
                )
                y -= int(0.6 * leading)

        # --- Paragraphs: use text_verbatim ---
        plist = sec.get("paragraphs") or []
        for pobj in plist:
            body = (pobj.get("text_verbatim") or "").strip()

            # Fallback: join claims text if text_verbatim missing
            if not body:
                claims = pobj.get("claims") or []
                joined = []
                for cl in claims:
                    t = (cl.get("text") or "").strip()
                    if t:
                        joined.append(t)
                body = "\n".join(joined).strip()

            if not body:
                continue

            # Page break guard
            if y < bottom + (leading * 3):
                y = new_page()

            # Clean ET WS mode: number every paragraph block
            if not analysis:
                body = f"{para_counter}. {body}"
                para_counter += 1

            y = draw_paragraph(
                c, body, left, y, max_w,
                font, size, leading,
                justify=bool(justify_body)
            )
            y -= int(0.7 * leading)

    c.save()


# =============================================================================
# Notebook-friendly wrapper (mirrors your CONFIG names)
# =============================================================================

def run_from_notebook(
    JSON_PATH: str,
    OUT_PDF: str,
    ANALYSIS: bool,
    *,
    PAGE_W_H: Tuple[float, float] = A4,
    LEFT: float = 2.0 * cm,
    RIGHT: float = 2.0 * cm,
    TOP: float = 2.0 * cm,
    BOTTOM: float = 2.0 * cm,
    FONT: str = "Times-Roman",
    FONT_B: str = "Times-Bold",
    SIZE: int = 11,
    LEADING: int = 14,
    JUSTIFY_BODY: bool = True,
    RENDER_SECTION_SUMMARY: bool = True,
) -> None:
    generate_ws_pdf(
        json_path=JSON_PATH,
        out_pdf=OUT_PDF,
        analysis=ANALYSIS,
        page_size=PAGE_W_H,
        left=LEFT,
        right=RIGHT,
        top=TOP,
        bottom=BOTTOM,
        font=FONT,
        font_bold=FONT_B,
        size=SIZE,
        leading=LEADING,
        justify_body=JUSTIFY_BODY,
        render_section_summary=RENDER_SECTION_SUMMARY,
    )
