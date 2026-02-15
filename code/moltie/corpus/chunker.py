from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Tuple, Dict, Optional

"""
Chunker layer: structural segmentation and span preservation.

Responsibilities:
- Split raw document text into stable paragraph units.
- Assign deterministic paragraph IDs (p00001, p00002, ...).
- Compute accurate (start_char, end_char) spans that round-trip to original text.
- Group paragraphs into retrieval-sized chunks (c00001, c00002, ...)
  with optional character overlap.

Guarantees:
- Paragraph spans slice back to the exact original text (modulo whitespace normalization).
- Paragraph ordering is deterministic.
- Chunk IDs are stable and ordered.
- Chunks preserve traceability to paragraph IDs.

Non-responsibilities:
- Does not interpret legal meaning.
- Does not rank relevance.
- Does not retrieve.
- Does not validate LLM outputs.

Output:
- List[Chunk] where Chunk = {doc_id, chunk_id, para_id, text, start_char, end_char}.
"""

_PARA_SPLIT = re.compile(
    r"\n\s*\n+"                    # blank lines
    r"|\n(?=\s*\(?\d+\)?[.)])"     # numbered para starts: 1. 1) (1)
)
  # blank-line paragraphs
_WS_RE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: str           # e.g. "c00012"
    para_id: str            # e.g. "p00123" within doc
    text: str
    start_char: int
    end_char: int


def _clean_para(s: str) -> str:
    s = s.strip()
    s = _WS_RE.sub(" ", s)
    return s.strip()


def iter_paragraphs(text: str) -> Iterator[Tuple[int, int, str]]:
    if not text:
        return

    def emit(block: str, start: int, end: int):
        cleaned = _clean_para(block)
        if cleaned:
            return (start, end, cleaned)
        return None

    # Build (start,end,raw) parts deterministically using split-boundaries.
    def split_with_spans(pattern: re.Pattern, s: str) -> List[Tuple[int, int, str]]:
        parts: List[Tuple[int, int, str]] = []
        prev_end = 0
        for m in pattern.finditer(s):
            seg = s[prev_end:m.start()]
            if seg.strip():
                parts.append((prev_end, m.start(), seg))
            prev_end = m.end()
        tail = s[prev_end:]
        if tail.strip():
            parts.append((prev_end, len(s), tail))
        return parts

    parts = split_with_spans(_PARA_SPLIT, text)

    # Fallback: if we got almost nothing, split on 1+ newlines but keep spans
    if len(parts) < 20:
        parts = split_with_spans(re.compile(r"\n+"), text)

    # Now pack small parts into paragraph-sized blocks (preserve true spans)
    buf = ""
    buf_start = None
    buf_end = None

    for s, e, raw in parts:
        raw_stripped = raw.strip()
        if not raw_stripped:
            continue

        if buf_start is None:
            buf_start = s
            buf_end = e
            buf = raw_stripped
            continue

        # candidate add
        candidate_len = len(buf) + 1 + len(raw_stripped)
        if candidate_len <= 900:
            buf = (buf + " " + raw_stripped).strip()
            buf_end = e
        else:
            out = emit(buf, buf_start, buf_end)
            if out:
                yield out
            buf_start = s
            buf_end = e
            buf = raw_stripped

    if buf and buf_start is not None and buf_end is not None:
        out = emit(buf, buf_start, buf_end)
        if out:
            yield out


def chunk_doc(
    doc_id: str,
    text: str,
    max_chars: int = 1400,
    overlap_chars: int = 150,
) -> List[Chunk]:
    """
    Produces chunks from paragraphs while preserving para_id.
    Para IDs are stable: p00001, p00002, ...
    Chunk IDs are stable order: c00001, c00002, ...
    """
    paras = list(iter_paragraphs(text))
    chunks: List[Chunk] = []

    # Build paragraph map with ids
    para_rows = []
    for i, (s, e, ptxt) in enumerate(paras, start=1):
        para_id = f"p{i:05d}"
        para_rows.append((para_id, s, e, ptxt))

    # Sliding accumulation of paragraphs into max_chars chunks
    buf: List[Tuple[str, int, int, str]] = []
    buf_len = 0
    chunk_idx = 0

    def flush_buffer(force: bool = False):
        nonlocal buf, buf_len, chunk_idx
        if not buf:
            return
        chunk_idx += 1
        chunk_id = f"c{chunk_idx:05d}"

        # chunk text is join of paragraphs
        start_char = buf[0][1]
        end_char = buf[-1][2]
        joined = "\n\n".join([b[3] for b in buf])

        # For anchor discipline we want the *most relevant* para_id easily referenced.
        # We store the FIRST para_id in the chunk as primary anchor id.
        primary_para_id = buf[0][0]

        chunks.append(Chunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            para_id=primary_para_id,
            text=joined,
            start_char=start_char,
            end_char=end_char,
        ))

        # overlap: keep tail chars worth of paragraphs
        if overlap_chars > 0 and not force:
            # keep paragraphs from the end until overlap covered
            keep: List[Tuple[str,int,int,str]] = []
            acc = 0
            for para in reversed(buf):
                acc += len(para[3])
                keep.append(para)
                if acc >= overlap_chars:
                    break
            keep = list(reversed(keep))
            buf = keep
            buf_len = sum(len(x[3]) for x in buf) + (2 * (len(buf)-1) if len(buf) > 1 else 0)
        else:
            buf = []
            buf_len = 0

    for row in para_rows:
        para_id, s, e, ptxt = row
        add_len = len(ptxt) + (2 if buf else 0)
        if buf and (buf_len + add_len) > max_chars:
            flush_buffer()
            # if overlap kept makes it still too big, clear overlap
            if buf and (buf_len + add_len) > max_chars:
                flush_buffer(force=True)  # clears buf
        buf.append(row)
        buf_len += add_len


    flush_buffer(force=True)
    return chunks
