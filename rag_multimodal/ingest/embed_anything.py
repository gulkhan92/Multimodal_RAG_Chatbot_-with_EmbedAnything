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
        # Map string model name to ea.WhichModel enum
        try:
            # This converts a string like "Bert" or "Clip" to ea.WhichModel.Bert or ea.WhichModel.Clip
            model_enum = getattr(ea.WhichModel, model_name)
        except AttributeError:
            raise ValueError(f"Model '{model_name}' is not a valid ea.WhichModel member. "
                             "Please check your .env and use values like 'Bert' or 'Clip'.")

        # from_pretrained_hf requires both a model_id (string) and the which_model enum.
        # We map common architecture names to their default Hugging Face model IDs.
        model_id_map = {
            "Bert": "sentence-transformers/all-MiniLM-L6-v2",
            "Clip": "openai/clip-vit-base-patch32",
        }
        
        model_id = model_id_map.get(model_name, "sentence-transformers/all-MiniLM-L6-v2")

        # The first argument 'model' expects a WhichModel enum, the second 'model_id' a string.
        self.embedder = ea.EmbeddingModel.from_pretrained_hf(model_enum, model_id)

        self.text_config: Optional[ea.TextEmbedConfig] = None

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # embed_query expects a list of strings (Vec<String> in Rust).
        # This returns a list of embedding objects.
        raw_vecs = ea.embed_query(texts, self.embedder, config=self.text_config)
        # Each 'v' is an Embedding object. We access the .embedding field which is the actual vector.
        return [_to_list_of_floats(np.asarray(v.embedding)) for v in raw_vecs]

    def embed_image(self, image_path: str) -> List[float]:
        """
        Embed a single image file.
        """
        p = Path(image_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(str(p))

        # embed_image typically expects a list of paths (Vec<String>) and returns a list of embeddings.
        # This matches the pattern seen in embed_query where a list was required.
        # embed_file returns a list of results. We index [0] to get the result for our single file.
        res = ea.embed_file(str(p), self.embedder)
        return _to_list_of_floats(np.asarray(res[0].embedding))
