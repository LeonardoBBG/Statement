from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

"""
Loader layer: document ingestion.

Responsibilities:
- Convert external storage formats (txt directories, jsonl files)
  into in-memory Doc objects.
- Preserve raw document text exactly as read.
- Attach lightweight metadata (source path, format, extra JSON fields).

Guarantees:
- Each Doc has a stable doc_id.
- Doc.text is the full raw text content (no chunking, no cleaning beyond strip()).
- No paragraph splitting, no retrieval, no reasoning.

Non-responsibilities:
- Does not parse PDFs.
- Does not split paragraphs.
- Does not score relevance.
- Does not modify text content structurally.

Output:
- List[Doc] where Doc = {doc_id, text, meta}.
"""

@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str
    meta: Dict[str, str]


def _read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def iter_docs_from_txt_dir(txt_dir: Path, glob: str = "*.txt") -> Iterator[Doc]:
    """
    Each .txt file becomes one Doc. doc_id = filename stem.
    """
    for p in sorted(txt_dir.glob(glob)):
        if not p.is_file():
            continue
        yield Doc(
            doc_id=p.stem,
            text=_read_text_file(p),
            meta={"source_path": str(p), "format": "txt"},
        )


def iter_docs_from_jsonl(jsonl_path: Path, text_key: str = "text", id_key: str = "doc_id") -> Iterator[Doc]:
    """
    JSONL lines: {doc_id: ..., text: ..., ...meta}
    """
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = str(obj.get(id_key) or obj.get("filename") or obj.get("uk_eat_no") or "NA").strip()
            text = str(obj.get(text_key) or "").strip()
            if not doc_id or not text:
                continue
            meta = {k: str(v) for k, v in obj.items() if k not in (text_key,)}
            meta["source_path"] = str(jsonl_path)
            meta["format"] = "jsonl"
            yield Doc(doc_id=doc_id, text=text, meta=meta)


def load_docs(
    sources: List[Path],
    mode: str,
    glob: str = "*.txt",
    text_key: str = "text",
    id_key: str = "doc_id",
) -> List[Doc]:
    """
    mode:
      - "txt_dir": each Path is a directory of .txt docs
      - "jsonl": each Path is a jsonl file
    """
    docs: List[Doc] = []
    if mode == "txt_dir":
        for d in sources:
            docs.extend(list(iter_docs_from_txt_dir(d, glob=glob)))
    elif mode == "jsonl":
        for p in sources:
            docs.extend(list(iter_docs_from_jsonl(p, text_key=text_key, id_key=id_key)))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return docs
