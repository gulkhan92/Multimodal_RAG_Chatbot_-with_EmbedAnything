from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional
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

    Supported methods (as used by this repo):
    - embed_text(texts)
    - embed_image(image_path)
    - embed_audio_file(audio_path)  # Whisper + text embedding underneath
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

        # Lazily initialized for audio embedding
        self._audio_decoder = None

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

        # embed_file returns a list of results. We index [0] to get the result for our single file.
        res = ea.embed_file(str(p), self.embedder)
        return _to_list_of_floats(np.asarray(res[0].embedding))

    def _get_audio_decoder(self, *, audio_decoder_model_id: str) -> Any:
        """
        Lazily load Whisper decoder.
        """
        if self._audio_decoder is None:
            self._audio_decoder = ea.AudioDecoderModel.from_pretrained_hf(
                audio_decoder_model_id
            )
        return self._audio_decoder

    def _get_text_embeder_for_audio(
        self,
        *,
        embedding_model_id: str,
    ) -> Any:
        """
        For audio embeddings, EmbedAnything can use a text embedding model.
        Reuse `self.embedder` when possible; otherwise load a new embedder.
        """
        # If the current embedder is already using the desired model_id, reuse.
        # The wrapper doesn't expose model_id directly, so we conservatively load a new one
        # only when the requested id differs from the default known mapping.
        # Current default mapping in __init__: sentence-transformers/all-MiniLM-L6-v2.
        if embedding_model_id == "sentence-transformers/all-MiniLM-L6-v2":
            return self.embedder
        return ea.EmbeddingModel.from_pretrained_hf(
            ea.WhichModel.Bert,
            embedding_model_id,
        )

    def embed_audio_file(
        self,
        audio_path: str,
        *,
        audio_decoder_model_id: str = "openai/whisper-tiny.en",
        embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 200,
        batch_size: int = 32,
    ) -> List[float]:
        """
        Embed a single audio file into a single vector.
        Implementation follows the EmbedAnything snippet using Whisper + text embedder.

        Returns: embedding vector (List[float])
        """
        p = Path(audio_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(str(p))

        audio_decoder = self._get_audio_decoder(audio_decoder_model_id=audio_decoder_model_id)
        embeder = self._get_text_embeder_for_audio(embedding_model_id=embedding_model_id)

        config = ea.TextEmbedConfig(chunk_size=chunk_size, batch_size=batch_size)

        # Use positional args to avoid keyword-name mismatches across embed_anything versions.
        data = ea.embed_audio_file(
            str(p),
            audio_decoder,
            embeder,
            config,
        )

        if not data:
            raise RuntimeError(f"embed_audio_file returned no embeddings for: {p}")

        # data is typically a list; take the first embedding vector
        first = data[0]
        emb = None
        if hasattr(first, "embedding"):
            emb = first.embedding
        elif isinstance(first, dict) and "embedding" in first:
            emb = first["embedding"]

        if emb is None:
            raise RuntimeError(
                "embed_audio_file returned an unexpected item format; expected .embedding"
            )

        return _to_list_of_floats(np.asarray(emb))
