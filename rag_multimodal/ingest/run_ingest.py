from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any, Dict, List

from rag_multimodal.ingest.chunking import chunk_text
from rag_multimodal.ingest.embed_anything import EmbedAnythingClient
from rag_multimodal.ingest.loader import discover_files
from rag_multimodal.ingest.media_extractor import extract_docx_images, extract_pdf_images_per_page
from rag_multimodal.ingest.pdf_extractor import extract_pdf_text_per_page
from rag_multimodal.ingest.text_extractor import extract_text_from_file
from rag_multimodal.ingest.doc_extractor import extract_text_from_docx
from rag_multimodal.ingest.quadrant_store import QuadrantStore
from rag_multimodal.settings import Settings


def _default_collection_for_modality(modality: str) -> str:
    return f"data_{modality}"


def _upsert_extracted_images(
    *,
    source_path: Path,
    images: List[Any],
    settings: Settings,
    store: QuadrantStore,
    source_modality: str,
    file_sha256: str | None = None,
) -> None:
    if not images:
        return

    store.collection = _default_collection_for_modality("png")
    embed_client = EmbedAnythingClient(model_name=settings.embedanything_image_model)

    ids: List[str] = []
    embeddings: List[List[float]] = []
    metadatas: List[Dict[str, Any]] = []

    for image in images:
        embedding = embed_client.embed_image(str(image.path))
        page_part = f"::p{image.page_index}" if image.page_index is not None else ""
        raw_id = f"{source_path.as_posix()}{page_part}::img{image.image_index}"
        image_id = f"img{image.image_index}"
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id)))
        embeddings.append(embedding)
        metadata: Dict[str, Any] = {
            "source_path": str(image.path),
            "file_name": image.path.name,
            "modality": "png",
            "source_modality": source_modality,
            "parent_source_path": str(source_path),
            "parent_file_name": source_path.name,
            "image_id": image_id,
        }
        if image.page_index is not None:
            metadata["page"] = image.page_index
        if file_sha256 is not None:
            metadata["file_sha256"] = file_sha256
        metadatas.append(metadata)

    store.upsert_embeddings(ids=ids, embeddings=embeddings, metadatas=metadatas)


# --- PDF Ingestion ---
def ingest_pdfs(*, data_dir: str | Path, settings: Settings, store: QuadrantStore) -> None:
    files = [d for d in discover_files(data_dir) if d.modality == "pdf"]
    if not files:
        print("No PDF files found.")
        return

    store.collection = _default_collection_for_modality("pdf")
    embed_client = EmbedAnythingClient(model_name=settings.embedanything_text_model)

    for f in files:
        store.collection = _default_collection_for_modality("pdf")
        print(f"Ingesting PDF: {f.path}")
        pages = extract_pdf_text_per_page(f.path)

        all_ids: List[str] = []
        all_embeddings: List[List[float]] = []
        all_metadatas: List[Dict[str, Any]] = []

        for page in pages:
            chunks = chunk_text(page.text)
            if not chunks:
                continue

            # Embed chunk texts
            # NOTE: EmbedAnythingClient.embed_text is currently a placeholder and must be wired.
            chunk_texts = [c.text for c in chunks]
            embeddings = embed_client.embed_text(chunk_texts)  # type: ignore[attr-defined]

            for c, emb in zip(chunks, embeddings):
                raw_id = f"{f.path.as_posix()}::p{page.page_index}::c{c.chunk_index}"
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))
                all_ids.append(chunk_id)
                all_embeddings.append(emb)
                all_metadatas.append(
                    {
                        "source_path": str(f.path),
                        "file_name": f.path.name,
                        "modality": "pdf",
                        "page": page.page_index,
                        "chunk_index": c.chunk_index,
                        "text": c.text,
                        "chunk_text_len": len(c.text),
                    }
                )

        if all_ids:
            store.upsert_embeddings(ids=all_ids, embeddings=all_embeddings, metadatas=all_metadatas)

        page_images = extract_pdf_images_per_page(f.path)
        if page_images:
            print(f"Ingesting {len(page_images)} image(s) extracted from PDF pages: {f.path}")
            _upsert_extracted_images(
                source_path=f.path,
                images=page_images,
                settings=settings,
                store=store,
                source_modality="pdf",
            )


# --- Generic Text File Ingestion (.txt, .md, .docx) ---
def ingest_text_files(*, data_dir: str | Path, settings: Settings, store: QuadrantStore) -> None:
    files = [d for d in discover_files(data_dir) if d.modality in {"txt", "md", "docx"}]
    if not files:
        print("No text files found.")
        return

    store.collection = _default_collection_for_modality("text") # Consolidate all text into 'data_text'
    embed_client = EmbedAnythingClient(model_name=settings.embedanything_text_model)

    for f in files:
        store.collection = _default_collection_for_modality("text")
        print(f"Ingesting Text File: {f.path}")
        
        file_content = ""
        if f.modality == "txt" or f.modality == "md":
            file_content = extract_text_from_file(f.path)
        elif f.modality == "docx":
            try:
                file_content = extract_text_from_docx(f.path)
            except ImportError as e:
                print(f"Skipping .docx file due to missing dependency: {e}")
                continue

        if not file_content.strip():
            print(f"Skipping empty or unextractable text from {f.path}")
            continue

        chunks = chunk_text(file_content)
        if not chunks:
            continue

        all_ids: List[str] = []
        all_embeddings: List[List[float]] = []
        all_metadatas: List[Dict[str, Any]] = []

        chunk_texts = [c.text for c in chunks]
        embeddings = embed_client.embed_text(chunk_texts)

        for c, emb in zip(chunks, embeddings):
            raw_id = f"{f.path.as_posix()}::c{c.chunk_index}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))
            all_ids.append(chunk_id)
            all_embeddings.append(emb)
            all_metadatas.append(
                {
                    "source_path": str(f.path),
                    "file_name": f.path.name,
                    "modality": f.modality, # Store original modality
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "chunk_text_len": len(c.text),
                }
            )
        if all_ids:
            store.upsert_embeddings(ids=all_ids, embeddings=all_embeddings, metadatas=all_metadatas)

        if f.modality == "docx":
            try:
                docx_images = extract_docx_images(f.path)
            except ImportError as e:
                print(f"Skipping .docx images due to missing dependency: {e}")
                docx_images = []

            if docx_images:
                print(f"Ingesting {len(docx_images)} image(s) extracted from DOCX: {f.path}")
                _upsert_extracted_images(
                    source_path=f.path,
                    images=docx_images,
                    settings=settings,
                    store=store,
                    source_modality="docx",
                )


def ingest_pngs(*, data_dir: str | Path, settings: Settings, store: QuadrantStore) -> None:
    files = [d for d in discover_files(data_dir) if d.modality == "png"]
    if not files:
        print("No PNG files found.")
        return

    store.collection = _default_collection_for_modality("png")
    embed_client = EmbedAnythingClient(model_name=settings.embedanything_image_model)

    for f in files:
        print(f"Ingesting PNG: {f.path}")

        # EmbedAnything image embedding placeholder.
        embedding = embed_client.embed_image(str(f.path))  # type: ignore[attr-defined]

        raw_id = f"{f.path.as_posix()}::img0"
        vec_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))
        store.upsert_embeddings(
            ids=[vec_id],
            embeddings=[embedding],
            metadatas=[
                {
                    "source_path": str(f.path),
                    "file_name": f.path.name,
                    "modality": "png",
                    "image_id": "img0",
                }
            ],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    settings = Settings.from_env()

    store = QuadrantStore(url=settings.quadrant_url, api_key=settings.quadrant_api_key)
    store.collection = _default_collection_for_modality("all")  # keep future-proof

    ingest_pdfs(data_dir=args.data_dir, settings=settings, store=store)
    ingest_text_files(data_dir=args.data_dir, settings=settings, store=store)
    ingest_pngs(data_dir=args.data_dir, settings=settings, store=store)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
