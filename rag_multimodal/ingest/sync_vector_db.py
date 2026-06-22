from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from rag_multimodal.ingest.chunking import chunk_text
from rag_multimodal.ingest.embed_anything import EmbedAnythingClient
from rag_multimodal.ingest.loader import discover_files
from rag_multimodal.ingest.media_extractor import extract_docx_images, extract_pdf_images_per_page
from rag_multimodal.ingest.text_extractor import extract_text_from_file
from rag_multimodal.ingest.doc_extractor import extract_text_from_docx
from rag_multimodal.ingest.pdf_extractor import extract_pdf_text_per_page
from rag_multimodal.ingest.quadrant_store import QuadrantStore
from rag_multimodal.settings import Settings


@dataclass(frozen=True)
class ManifestEntry:
    sha256: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(manifest_path: Path) -> Dict[str, ManifestEntry]:
    if not manifest_path.exists():
        return {}
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: Dict[str, ManifestEntry] = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "sha256" in v:
            out[k] = ManifestEntry(sha256=str(v["sha256"]))
    return out


def _save_manifest(manifest_path: Path, data: Dict[str, ManifestEntry]) -> None:
    payload: Dict[str, Any] = {k: {"sha256": v.sha256} for k, v in data.items()}
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _file_key(data_dir: Path, file_path: Path) -> str:
    # stable key across runs
    return file_path.relative_to(data_dir).as_posix()


def ingest_changed_pdfs(
    *,
    data_dir: Path,
    settings: Settings,
    store: QuadrantStore,
    embed_client: EmbedAnythingClient,
    image_embed_client: EmbedAnythingClient,
    manifest: Dict[str, ManifestEntry],
) -> Dict[str, ManifestEntry]:
    files = [d for d in discover_files(data_dir) if d.modality == "pdf"]
    if not files:
        print("No PDF files found.")
        return manifest
    store.collection = "data_text" # Consolidate all text into a single collection

    for f in files:
        key = _file_key(data_dir, f.path)
        sha = _sha256_file(f.path)
        entry = manifest.get(key)

        if entry and entry.sha256 == sha:
            continue  # unchanged

        print(f"[sync] Ingesting changed PDF: {f.path}")

        pages = extract_pdf_text_per_page(f.path)
        for_upsert_ids: List[str] = []
        for_upsert_embeddings: List[List[float]] = []
        for_upsert_metadatas: List[Dict[str, Any]] = []

        for page in pages:
            chunks = chunk_text(page.text)
            if not chunks:
                continue

            chunk_texts = [c.text for c in chunks]
            embeddings = embed_client.embed_text(chunk_texts)

            for c, emb in zip(chunks, embeddings):
                raw_id = f"{f.path.as_posix()}::p{page.page_index}::c{c.chunk_index}"
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))
                for_upsert_ids.append(chunk_id)
                for_upsert_embeddings.append(emb)
                for_upsert_metadatas.append(
                    {
                        "source_path": str(f.path),
                        "file_name": f.path.name,
                        "modality": "pdf",
                        "page": page.page_index,
                        "chunk_index": c.chunk_index,
                        "text": c.text,
                        "chunk_text_len": len(c.text),
                        "file_sha256": sha,
                    }
                )

        # True sync: delete any previous vectors for this file before upserting new ones.
        store.delete_by_filter(
            filter_metadata={
                "source_path": str(f.path),
                "modality": "pdf",
            }
        )

        if for_upsert_ids:
            store.upsert_embeddings(
                ids=for_upsert_ids,
                embeddings=for_upsert_embeddings,
                metadatas=for_upsert_metadatas,
            )

        page_images = extract_pdf_images_per_page(f.path)
        store.collection = "data_png"
        store.delete_by_filter(
            filter_metadata={
                "parent_source_path": str(f.path),
                "source_modality": "pdf",
            }
        )

        if page_images:
            image_ids: List[str] = []
            image_embeddings: List[List[float]] = []
            image_metadatas: List[Dict[str, Any]] = []
            for image in page_images:
                raw_id = f"{f.path.as_posix()}::p{image.page_index}::img{image.image_index}"
                image_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id)))
                image_embeddings.append(image_embed_client.embed_image(str(image.path)))
                image_metadatas.append(
                    {
                        "source_path": str(image.path),
                        "file_name": image.path.name,
                        "modality": "png",
                        "source_modality": "pdf",
                        "parent_source_path": str(f.path),
                        "parent_file_name": f.path.name,
                        "page": image.page_index,
                        "image_id": f"img{image.image_index}",
                        "file_sha256": sha,
                    }
                )

            store.upsert_embeddings(
                ids=image_ids,
                embeddings=image_embeddings,
                metadatas=image_metadatas,
            )

        if for_upsert_ids or page_images:
            manifest[key] = ManifestEntry(sha256=sha)

    return manifest


def ingest_changed_text_files(
    *,
    data_dir: Path,
    settings: Settings,
    store: QuadrantStore,
    embed_client: EmbedAnythingClient,
    image_embed_client: EmbedAnythingClient,
    manifest: Dict[str, ManifestEntry],
) -> Dict[str, ManifestEntry]:
    files = [d for d in discover_files(data_dir) if d.modality in {"txt", "md", "docx"}]
    if not files:
        print("No text files found.")
        return manifest
    store.collection = "data_text"  # Consolidate all text into 'data_text'

    for f in files:
        store.collection = "data_text"
        key = _file_key(data_dir, f.path)
        sha = _sha256_file(f.path)
        entry = manifest.get(key)

        if entry and entry.sha256 == sha:
            continue  # unchanged

        print(f"[sync] Ingesting changed Text File: {f.path}")

        file_content = ""
        if f.modality == "txt" or f.modality == "md":
            file_content = extract_text_from_file(f.path)
        elif f.modality == "docx":
            try:
                file_content = extract_text_from_docx(f.path)
            except ImportError as e:
                print(f"Skipping .docx file due to missing dependency: {e}")
                continue

        docx_images = []
        if f.modality == "docx":
            try:
                docx_images = extract_docx_images(f.path)
            except ImportError as e:
                print(f"Skipping .docx images due to missing dependency: {e}")

        if not file_content.strip() and not docx_images:
            print(f"Skipping empty or unextractable text from {f.path}")
            continue

        chunks = chunk_text(file_content)
        for_upsert_ids: List[str] = []
        for_upsert_embeddings: List[List[float]] = []
        for_upsert_metadatas: List[Dict[str, Any]] = []

        if chunks:
            chunk_texts = [c.text for c in chunks]
            embeddings = embed_client.embed_text(chunk_texts)

            for c, emb in zip(chunks, embeddings):
                raw_id = f"{f.path.as_posix()}::c{c.chunk_index}"
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))
                for_upsert_ids.append(chunk_id)
                for_upsert_embeddings.append(emb)
                for_upsert_metadatas.append(
                    {
                        "source_path": str(f.path),
                        "file_name": f.path.name,
                        "modality": f.modality,
                        "chunk_index": c.chunk_index,
                        "text": c.text,
                        "chunk_text_len": len(c.text),
                        "file_sha256": sha,
                    }
                )

        store.delete_by_filter(filter_metadata={"source_path": str(f.path), "modality": f.modality})

        if for_upsert_ids:
            store.upsert_embeddings(ids=for_upsert_ids, embeddings=for_upsert_embeddings, metadatas=for_upsert_metadatas)

        if f.modality == "docx":
            store.collection = "data_png"
            store.delete_by_filter(
                filter_metadata={
                    "parent_source_path": str(f.path),
                    "source_modality": "docx",
                }
            )

        if docx_images:
            image_ids: List[str] = []
            image_embeddings: List[List[float]] = []
            image_metadatas: List[Dict[str, Any]] = []
            for image in docx_images:
                raw_id = f"{f.path.as_posix()}::img{image.image_index}"
                image_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id)))
                image_embeddings.append(image_embed_client.embed_image(str(image.path)))
                image_metadatas.append(
                    {
                        "source_path": str(image.path),
                        "file_name": image.path.name,
                        "modality": "png",
                        "source_modality": "docx",
                        "parent_source_path": str(f.path),
                        "parent_file_name": f.path.name,
                        "image_id": f"img{image.image_index}",
                        "file_sha256": sha,
                    }
                )

            store.upsert_embeddings(
                ids=image_ids,
                embeddings=image_embeddings,
                metadatas=image_metadatas,
            )

        if for_upsert_ids or docx_images:
            manifest[key] = ManifestEntry(sha256=sha)

    return manifest


def ingest_changed_pngs(
    *,
    data_dir: Path,
    settings: Settings,
    store: QuadrantStore,
    embed_client: EmbedAnythingClient,
    manifest: Dict[str, ManifestEntry],
) -> Dict[str, ManifestEntry]:
    files = [d for d in discover_files(data_dir) if d.modality == "png"]
    if not files:
        print("No PNG files found.")
        return manifest

    store.collection = "data_png"

    for f in files:
        key = _file_key(data_dir, f.path)
        sha = _sha256_file(f.path)
        entry = manifest.get(key)

        if entry and entry.sha256 == sha:
            continue  # unchanged

        print(f"[sync] Ingesting changed PNG: {f.path}")

        # True sync: delete any previous vectors for this file before upserting new ones.
        store.delete_by_filter(
            filter_metadata={
                "source_path": str(f.path),
                "modality": "png",
            }
        )

        embedding = embed_client.embed_image(str(f.path))

        raw_id = f"{f.path.as_posix()}::img0"
        vec_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))

        store.upsert_embeddings(
            ids=[vec_id],
            embeddings=[embedding],
            metadatas=[
                {
                    "source_path": str(f.path),
                    "file_name": f.path.name,
                    "modality": "png",  # This was missing
                    "image_id": "img0",
                    "file_sha256": sha,
                }
            ],
        )

        manifest[key] = ManifestEntry(sha256=sha)

    return manifest


def ingest_changed_audios(
    *,
    data_dir: Path,
    settings: Settings,
    store: QuadrantStore,
    embed_client: EmbedAnythingClient,
    manifest: Dict[str, ManifestEntry],
) -> Dict[str, ManifestEntry]:
    """
    Incrementally sync supported audio files into Quadrant.
    Audio discovery is handled by loader.discover_files(...), which maps
    multiple audio extensions to modality="audio".
    """
    files = [d for d in discover_files(data_dir) if d.modality == "audio"]
    if not files:
        print("No audio files found.")
        return manifest

    store.collection = "data_audio"

    for f in files:
        key = _file_key(data_dir, f.path)
        sha = _sha256_file(f.path)
        entry = manifest.get(key)

        if entry and entry.sha256 == sha:
            continue  # unchanged

        print(f"[sync] Ingesting changed Audio: {f.path}")

        # True sync: delete any previous vectors for this file before upserting new ones.
        store.delete_by_filter(
            filter_metadata={
                "source_path": str(f.path),
                "modality": "audio",
            }
        )

        embedding = embed_client.embed_audio_file(str(f.path))

        # single-vector per file for MVP
        raw_id = f"{f.path.as_posix()}::audio0"
        vec_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))

        store.upsert_embeddings(
            ids=[vec_id],
            embeddings=[embedding],
            metadatas=[
                {
                    "source_path": str(f.path),
                    "file_name": f.path.name,
                    "modality": "audio",
                    "audio_id": "audio0",
                    "file_sha256": sha,
                }
            ],
        )

        manifest[key] = ManifestEntry(sha256=sha)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incrementally sync Quadrant from files under data-dir (PDF + PNG)."
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Folder containing input documents.")
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Optional Quadrant collection name. If omitted, a default is used.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=".rag_ingest_manifest.json",
        help="Local manifest file to track ingested file hashes.",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    data_dir = Path(args.data_dir)

    store_kwargs: Dict[str, Any] = {"api_key": settings.quadrant_api_key}
    if settings.quadrant_url:
        store_kwargs["url"] = settings.quadrant_url
    if settings.quadrant_path:
        store_kwargs["path"] = settings.quadrant_path

    store = QuadrantStore(**store_kwargs)

    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)

    text_client = EmbedAnythingClient(model_name=settings.embedanything_text_model)
    image_client = EmbedAnythingClient(model_name=settings.embedanything_image_model)

    manifest = ingest_changed_pdfs(
        data_dir=data_dir,
        settings=settings,
        store=store,
        embed_client=text_client,
        image_embed_client=image_client,
        manifest=manifest,
    )
    manifest = ingest_changed_text_files(
        data_dir=data_dir,
        settings=settings,
        store=store,
        embed_client=text_client, # Use text_client for all text modalities
        image_embed_client=image_client,
        manifest=manifest,
    )
    manifest = ingest_changed_pngs(
        data_dir=data_dir,
        settings=settings,
        store=store,
        embed_client=image_client,
        manifest=manifest,
    )

    # Audio embeddings (Whisper -> text embedding -> vector)
    audio_client = EmbedAnythingClient(model_name=settings.embedanything_text_model)
    manifest = ingest_changed_audios(
        data_dir=data_dir,
        settings=settings,
        store=store,
        embed_client=audio_client,
        manifest=manifest,
    )

    _save_manifest(manifest_path, manifest)
    print("Sync complete.")


if __name__ == "__main__":
    main()
