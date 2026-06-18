from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class DiscoveredFile:
    path: Path
    modality: str  # "pdf" or "png"


def discover_files(data_dir: str | os.PathLike) -> List[DiscoveredFile]:
    """
    Discover supported files under data_dir (recursively).
    Returned `modality` must match the ingest functions' expectations.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data-dir does not exist: {root}")

    out: List[DiscoveredFile] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue

        suffix = p.suffix.lower().lstrip(".")

        # Text modalities (used by ingest_changed_text_files)
        if suffix in {"txt"}:
            out.append(DiscoveredFile(path=p, modality="txt"))
            continue
        if suffix in {"md", "markdown"}:
            out.append(DiscoveredFile(path=p, modality="md"))
            continue
        if suffix in {"docx"}:
            out.append(DiscoveredFile(path=p, modality="docx"))
            continue

        # PDF
        if suffix == "pdf":
            out.append(DiscoveredFile(path=p, modality="pdf"))
            continue

        # Audio: map all supported audio formats to modality "audio"
        if suffix in {"wav", "mp3", "m4a", "aac", "flac", "ogg", "oga", "opus", "wma"}:
            out.append(DiscoveredFile(path=p, modality="audio"))
            continue

        # Images: normalize all raster formats to modality "png" so sync can embed them.
        if suffix in {"png", "jpg", "jpeg", "webp", "bmp", "gif", "tiff", "tif"}:
            out.append(DiscoveredFile(path=p, modality="png"))
            continue

    return out
