from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from pypdf import PdfReader

try:
    from docx import Document
except ImportError:
    Document = None


@dataclass(frozen=True)
class ExtractedImage:
    path: Path
    image_index: int
    page_index: Optional[int] = None


_CONTENT_TYPE_EXTENSIONS: Dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "document"


def _image_output_dir(source_path: Path, output_root: str | Path) -> Path:
    out_dir = Path(output_root) / _safe_stem(source_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _extension_from_name(name: str, default: str = ".png") -> str:
    suffix = Path(name).suffix.lower()
    return suffix if suffix else default


def extract_pdf_images_per_page(
    pdf_path: str | Path,
    *,
    output_root: str | Path = ".rag_extracted_images",
) -> List[ExtractedImage]:
    reader = PdfReader(str(pdf_path))
    source_path = Path(pdf_path)
    out_dir = _image_output_dir(source_path, output_root)
    extracted: List[ExtractedImage] = []

    for page_index, page in enumerate(reader.pages):
        for image_index, image in enumerate(getattr(page, "images", []) or []):
            image_name = getattr(image, "name", f"image_{image_index}.png")
            ext = _extension_from_name(image_name)
            out_path = out_dir / f"page_{page_index}_image_{image_index}{ext}"
            out_path.write_bytes(image.data)
            extracted.append(
                ExtractedImage(path=out_path, page_index=page_index, image_index=image_index)
            )

    return extracted


def extract_docx_images(
    docx_path: str | Path,
    *,
    output_root: str | Path = ".rag_extracted_images",
) -> List[ExtractedImage]:
    if Document is None:
        raise ImportError("python-docx is required to process .docx files.")

    source_path = Path(docx_path)
    doc = Document(source_path)
    out_dir = _image_output_dir(source_path, output_root)
    extracted: List[ExtractedImage] = []

    image_parts = []
    for rel in doc.part.rels.values():
        try:
            target = rel.target_part
        except ValueError:
            continue
        if getattr(target, "content_type", "").startswith("image/"):
            image_parts.append(target)

    for image_index, image_part in enumerate(image_parts):
        content_type = getattr(image_part, "content_type", "")
        ext = _CONTENT_TYPE_EXTENSIONS.get(content_type, ".png")
        out_path = out_dir / f"doc_image_{image_index}{ext}"
        out_path.write_bytes(image_part.blob)
        extracted.append(ExtractedImage(path=out_path, image_index=image_index))

    return extracted
