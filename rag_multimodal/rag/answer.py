from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rag_multimodal.ingest.embed_anything import EmbedAnythingClient
from rag_multimodal.ingest.quadrant_store import RetrievedChunk, QuadrantStore
from rag_multimodal.rag.gemini_client import GeminiClient
from rag_multimodal.rag.prompt import build_gemini_prompt
from rag_multimodal.rag.retrieve import retrieve


@dataclass(frozen=True)
class RAGResponse:
    answer: str
    citations: List[int]


def parse_citations(text: str) -> List[int]:
    """
    Extract citations like [0], [3] from the Gemini response.
    """
    import re

    idxs = re.findall(r"\[(\d+)\]", text)
    out: List[int] = []
    for s in idxs:
        try:
            out.append(int(s))
        except ValueError:
            continue
    # de-dup preserving order
    seen = set()
    deduped: List[int] = []
    for i in out:
        if i not in seen:
            seen.add(i)
            deduped.append(i)
    return deduped


def answer_question(
    *,
    question: str,
    settings,
    embed_client: EmbedAnythingClient,
    store: QuadrantStore,
    gemini: GeminiClient,
    top_k: int = 8,
) -> RAGResponse:
    retrieval = retrieve(query=question, embed_client=embed_client, store=store, top_k=top_k)
    chunks: List[RetrievedChunk] = retrieval.chunks

    prompt = build_gemini_prompt(question=question, chunks=chunks)
    gemini_resp = gemini.generate(prompt=prompt)

    citations = parse_citations(gemini_resp.text)
    return RAGResponse(answer=gemini_resp.text.strip(), citations=citations)
