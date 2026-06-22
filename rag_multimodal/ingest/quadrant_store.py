from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    VectorParams,
    PointStruct,
)


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    score: float
    metadata: Dict[str, Any]
    text: Optional[str] = None


class QuadrantStore:
    """
    Qdrant-backed vector store.

    Your project calls it "Quadrant" but the Python dependency is `qdrant-client`.
    """

    def __init__(
        self,
        *,
        url: str = "",
        path: str = "",
        api_key: str = "",
        collection: str = "data_multimodal",
    ):
        self.url = url
        self.path = path
        self.api_key = api_key
        self.collection = collection

        # Note: we persist locally when `path` is set (QdrantClient(path=...)).
        #
        # Qdrant local mode uses file locking. If another process already created
        # a local client pointing at the same folder, QdrantClient(path=...)
        # will raise a RuntimeError/AlreadyLocked. In that case, fall back to
        # server mode (url) if available, so the API can still start.
        if path:
            # Persistent local Qdrant storage.
            # If this fails due to a lock, it will raise an exception, which is
            # the desired behavior to prevent multiple writers.
            self._client = QdrantClient(path=path)
        else:
            if not url:
                raise RuntimeError("QuadrantStore requires either `url` or `path`.")
            self._client = QdrantClient(url=url, api_key=api_key)

    def _collection_exists(self) -> bool:
        collections = self._client.get_collections().collections
        return any(c.name == self.collection for c in collections)

    def _ensure_collection(self, *, vector_size: int) -> None:
        if self._collection_exists():
            return

        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert_embeddings(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, Any]],
    ) -> None:
        """
        Upsert vectors into Qdrant.

        ids, embeddings, metadatas must be parallel arrays.
        """
        if not ids:
            return

        if not (len(ids) == len(embeddings) == len(metadatas)):
            raise ValueError("ids, embeddings, metadatas must have the same length")

        vector_size = len(embeddings[0])
        self._ensure_collection(vector_size=vector_size)

        points: List[PointStruct] = []
        for _id, vec, meta in zip(ids, embeddings, metadatas):
            points.append(
                PointStruct(
                    id=_id,
                    vector=list(vec),
                    payload=dict(meta),
                )
            )

        self._client.upsert(collection_name=self.collection, points=points)

    def _build_filter(self, *, filter_metadata: Dict[str, Any]) -> Filter:
        conditions: List[FieldCondition] = []
        for k, v in filter_metadata.items():
            conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
        return Filter(must=conditions)

    def delete_by_filter(self, *, filter_metadata: Dict[str, Any]) -> None:
        """
        Delete all points where payload matches all provided key/value pairs.
        """
        if not filter_metadata:
            return
        if not self._collection_exists():
            return
        flt = self._build_filter(filter_metadata=filter_metadata)
        self._client.delete(collection_name=self.collection, points_selector=flt)

    def count_by_filter(self, *, filter_metadata: Dict[str, Any]) -> int:
        """
        Return how many points match the payload filter.
        """
        if not filter_metadata:
            return 0
        if not self._collection_exists():
            return 0
        flt = self._build_filter(filter_metadata=filter_metadata)
        res = self._client.count(collection_name=self.collection, count_filter=flt)
        return int(res.count)

    def similarity_search(
        self,
        *,
        embedding: Sequence[float],
        top_k: int = 8,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        Search Qdrant for similar items using vector cosine similarity.
        """
        if not embedding:
            return []

        if not self._collection_exists():
            return []

        query_filter: Optional[Filter] = None
        if filter_metadata:
            # Build an "AND" filter: all key/value pairs must match.
            conditions: List[FieldCondition] = []
            for k, v in filter_metadata.items():
                conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
            query_filter = Filter(must=conditions)

        # Qdrant client uses `query_points` in this SDK version.
        res = self._client.query_points(
            collection_name=self.collection,
            query=embedding,  # type: ignore[arg-type]
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        # `query_points` returns QueryResponse with `points` list of ScoredPoint
        out: List[RetrievedChunk] = []
        for p in res.points or []:
            payload = p.payload or {}
            out.append(
                RetrievedChunk(
                    id=str(p.id),
                    score=float(p.score),
                    metadata=dict(payload),
                    text=payload.get("text"),
                )
            )
        return out
