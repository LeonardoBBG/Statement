from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, List, Tuple, Dict, Optional


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
    """
    Yield (start_char, end_char, paragraph_text) using blank-line splitting.
    Keeps positions for traceability.
    """
    if not text:
        return
    # We need positions: do manual scan
    parts = _PARA_SPLIT.split(text)
    cursor = 0
    for part in parts:
        if not part:
            continue
        # find this part in original text near cursor
        idx = text.find(part, cursor)
        if idx == -1:
            # fallback: approximate
            idx = cursor
        start = idx
        end = idx + len(part)
        cursor = end
        cleaned = _clean_para(part)
        if cleaned:
            yield start, end, cleaned


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
        buf.append(row)
        buf_len += add_len

    flush_buffer(force=True)
    return chunks
