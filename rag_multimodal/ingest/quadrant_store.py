from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: F401

# Quadrant SDK/client may differ depending on version.
# This wrapper isolates Quadrant-specific code to make it easy to adjust.


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    score: float
    metadata: Dict[str, Any]
    text: Optional[str] = None


class QuadrantStore:
    def __init__(
        self,
        *,
        url: str,
        api_key: str = "",
        collection: str = "data_multimodal",
    ):
        self.url = url
        self.api_key = api_key
        self.collection = collection

        # TODO: Replace with real Quadrant client initialization after verifying SDK.
        self._client = None

    def upsert_embeddings(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        """
        Upsert vectors into Quadrant.

        IDs and embeddings are parallel arrays.
        """
        raise NotImplementedError(
            "QuadrantStore.upsert_embeddings is not implemented yet. "
            "Update this file with Quadrant's real upsert API."
        )

    def similarity_search(
        self,
        *,
        embedding: Sequence[float],
        top_k: int = 8,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        Search Quadrant for similar items.
        """
        raise NotImplementedError(
            "QuadrantStore.similarity_search is not implemented yet. "
            "Update this file with Quadrant's real search API."
        )
