from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from pypdf import PdfReader


@dataclass(frozen=True)
class PdfPageText:
    page_index: int
    text: str


def extract_pdf_text_per_page(pdf_path: str | Path) -> List[PdfPageText]:
    reader = PdfReader(str(pdf_path))
    out: List[PdfPageText] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        out.append(PdfPageText(page_index=i, text=text))
    return out
