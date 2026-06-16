from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rag_multimodal.ingest.quadrant_store import RetrievedChunk


@dataclass(frozen=True)
class PromptBlock:
    label: str
    content: str


def build_gemini_prompt(*, question: str, chunks: List[RetrievedChunk]) -> str:
    """
    Build a grounded prompt with citations.

    Gemini will be instructed to use ONLY provided context.
    """
    context_parts: List[str] = []
    for i, ch in enumerate(chunks):
        meta = ch.metadata or {}
        source = meta.get("source_path") or meta.get("file_name") or "unknown_source"
        page = meta.get("page", None)
        chunk_index = meta.get("chunk_index", None)
        modality = meta.get("modality", "unknown")

        header_bits = [f"[{i}] modality={modality}", f"source={source}"]
        if page is not None:
            header_bits.append(f"page={page}")
        if chunk_index is not None:
            header_bits.append(f"chunk_index={chunk_index}")
        header = " ".join(header_bits)

        text = ch.text or meta.get("text") or ""
        text = (text or "").strip()
        if not text:
            # For image chunks we may not have text; still include a placeholder.
            text = f"(no extracted text available for this chunk)"

        context_parts.append(f"{header}\n{indent(text, 2)}")

    context = "\n\n".join(context_parts)
    return (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the context is insufficient, say you don't know.\n\n"
        "Context:\n"
        f"{context}\n\n"
        f"User question: {question}\n\n"
        "Answer format:\n"
        "- Answer: <your grounded answer>\n"
        "- Citations: list the [i] indices you used (e.g. [0], [3])"
    )


def indent(s: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in s.splitlines())
