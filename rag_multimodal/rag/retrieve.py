from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rag_multimodal.ingest.embed_anything import EmbedAnythingClient
from rag_multimodal.ingest.quadrant_store import QuadrantStore, RetrievedChunk


@dataclass(frozen=True)
class RetrievalResult:
    chunks: List[RetrievedChunk]


def retrieve(
    *,
    query: str,
    embed_client: EmbedAnythingClient,
    store: QuadrantStore,
    top_k: int = 8,
) -> RetrievalResult:
    """
    Query embedding -> Quadrant similarity search -> topK chunks
    """
    query_embedding = embed_client.embed_text([query])[0]  # type: ignore[attr-defined]

    results = store.similarity_search(
        embedding=query_embedding,
        top_k=top_k,
        filter_metadata=None,
    )

    # store.similarity_search returns RetrievedChunk already
    return RetrievalResult(chunks=results)
