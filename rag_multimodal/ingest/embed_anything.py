from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

import numpy as np

import embed_anything as ea


@dataclass(frozen=True)
class EmbeddedItem:
    embedding: List[float]


def _to_list_of_floats(arr: np.ndarray) -> List[float]:
    return [float(x) for x in np.asarray(arr).reshape(-1).tolist()]


class EmbedAnythingClient:
    """
    Wrapper over the installed `embed_anything` package.

    Discovered signatures:
    - embed_query(query, embedder, config=None)
    - embed_image_directory(directory, embedder, config=None, adapter=None)
    """

    def __init__(self, model_name: str = ""):
        self.model_name = model_name

        # Instantiate an embedder; constructor signature may vary by version.
        try:
            self.embedder = ea.EmbeddingModel(model_name=model_name)  # type: ignore[call-arg]
        except TypeError:
            self.embedder = ea.EmbeddingModel(model_name)  # type: ignore[call-arg]

        self.text_config: Optional[ea.TextEmbedConfig] = None

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        vectors: List[List[float]] = []
        for t in texts:
            vec = ea.embed_query(t, self.embedder, config=self.text_config)
            vectors.append(_to_list_of_floats(np.asarray(vec)))
        return vectors

    def embed_image(self, image_path: str) -> List[float]:
        """
        MVP embedding approach:
        - call `embed_image_directory()` on the parent directory
        - require the parent directory to contain exactly one image file
          so we can select the correct embedding deterministically
        """
        p = Path(image_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(str(p))

        parent = p.parent
        images = [x for x in parent.iterdir() if x.is_file() and x.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        if len(images) != 1:
            raise RuntimeError(
                "EmbedAnythingClient.embed_image MVP expects the image parent directory to contain exactly one image. "
                f"Parent dir has {len(images)} images: {[x.name for x in images]}"
            )

        vecs = ea.embed_image_directory(
            str(parent),
            self.embedder,
            config=None,
            adapter=None,
        )

        # embed_image_directory returns embeddings in file order; since there's 1 file, take the first.
        return _to_list_of_floats(np.asarray(vecs[0]))
