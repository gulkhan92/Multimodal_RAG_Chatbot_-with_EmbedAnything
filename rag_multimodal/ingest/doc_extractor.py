from __future__ import annotations

from pathlib import Path

try:
    from docx import Document
except ImportError:
    print("python-docx not installed. Cannot process .docx files.")
    Document = None


def extract_text_from_docx(file_path: Path) -> str:
    if Document is None:
        raise ImportError("python-docx is required to process .docx files.")
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])