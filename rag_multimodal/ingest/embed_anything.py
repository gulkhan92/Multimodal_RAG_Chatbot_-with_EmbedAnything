from __future__ import annotations

from dataclasses import dataclass
from typing import List

# NOTE:
# EmbedAnything's Python import path and embedding API may differ depending on the package version.
# This wrapper is intentionally isolated so you can adjust it quickly once you validate locally.

import numpy as np


@dataclass(frozen=True)
class EmbeddedItem:
    embedding: List[float]


def _to_list_of_floats(arr: np.ndarray) -> List[float]:
    return [float(x) for x in arr.reshape(-1).tolist()]


class EmbedAnythingClient:
    def __init__(self, model_name: str = ""):
        self.model_name = model_name

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """
        Returns an embedding per text item.

        You must update this method once you confirm EmbedAnything's real Python API.
        """
        if not texts:
            return []
        raise NotImplementedError(
            "EmbedAnythingClient.embed_text is not implemented yet. "
            "Update this file with EmbedAnything's real API."
        )

    def embed_image(self, image_path: str) -> List[float]:
        """
        Optional for MVP; may be implemented later if you want OCR fallback vs raw image embedding.
        """
        raise NotImplementedError(
            "EmbedAnythingClient.embed_image is not implemented yet. "
            "Update this file with EmbedAnything's real API."
        )
