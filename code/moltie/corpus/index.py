from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

from .loader import Doc
from .chunker import Chunk, chunk_doc


@dataclass
class CorpusIndex:
    docs: Dict[str, Doc]
    chunks_by_doc: Dict[str, List[Chunk]]

    def get_doc(self, doc_id: str) -> Doc:
        return self.docs[doc_id]

    def get_chunks(self, doc_id: str) -> List[Chunk]:
        return self.chunks_by_doc.get(doc_id, [])


def build_corpus_index(
    docs: List[Doc],
    max_chars: int = 1400,
    overlap_chars: int = 150,
) -> CorpusIndex:
    docs_map: Dict[str, Doc] = {}
    chunks_map: Dict[str, List[Chunk]] = {}

    for d in docs:
        docs_map[d.doc_id] = d
        chunks_map[d.doc_id] = chunk_doc(d.doc_id, d.text, max_chars=max_chars, overlap_chars=overlap_chars)

    return CorpusIndex(docs=docs_map, chunks_by_doc=chunks_map)


def keyword_search_chunks(
    index: CorpusIndex,
    terms: List[str],
    top_k_docs: int = 200,
    top_k_chunks_per_doc: int = 12,
) -> List[Tuple[str, float, List[Chunk]]]:
    """
    Cheap baseline retrieval:
    - score chunks by keyword hit count
    - aggregate to doc score = sum top chunk scores
    Returns: [(doc_id, doc_score, top_chunks)]
    """
    terms_n = [t.strip().lower() for t in terms if t and t.strip()]
    if not terms_n:
        return []

    results: List[Tuple[str, float, List[Chunk]]] = []

    for doc_id, chunks in index.chunks_by_doc.items():
        scored: List[Tuple[float, Chunk]] = []
        for ch in chunks:
            t = ch.text.lower()
            hits = 0
            for term in terms_n:
                if term and term in t:
                    hits += 1
            if hits:
                scored.append((float(hits), ch))

        if not scored:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c for _, c in scored[:top_k_chunks_per_doc]]
        doc_score = sum(s for s, _ in scored[:top_k_chunks_per_doc])
        results.append((doc_id, doc_score, top_chunks))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k_docs]
