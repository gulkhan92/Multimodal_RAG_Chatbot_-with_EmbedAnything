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
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"data-dir does not exist: {root}")

    out: List[DiscoveredFile] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower().lstrip(".")
        if suffix == "pdf":
            out.append(DiscoveredFile(path=p, modality="pdf"))
        elif suffix in {"png"}:
            out.append(DiscoveredFile(path=p, modality="png"))
    return out
