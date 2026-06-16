from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    text: str


def chunk_text(text: str, *, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[TextChunk]:
    """
    Simple char-based chunking to keep MVP dependencies low.

    For PDFs we chunk extracted text; later you can replace with token-based chunking.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[TextChunk] = []
    start = 0
    idx = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        part = cleaned[start:end].strip()
        if part:
            chunks.append(TextChunk(chunk_index=idx, text=part))
            idx += 1
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if end == len(cleaned):
            break
    return chunks
