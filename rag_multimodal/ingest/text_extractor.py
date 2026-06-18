from __future__ import annotations

from pathlib import Path


def extract_text_from_file(file_path: Path) -> str:
    """
    Extracts text content from a plain text or markdown file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()