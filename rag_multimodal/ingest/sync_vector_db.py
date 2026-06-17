from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from rag_multimodal.ingest.chunking import chunk_text
from rag_multimodal.ingest.embed_anything import EmbedAnythingClient
from rag_multimodal.ingest.loader import discover_files
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
    manifest: Dict[str, ManifestEntry],
) -> Dict[str, ManifestEntry]:
    files = [d for d in discover_files(data_dir) if d.modality == "pdf"]
    if not files:
        print("No PDF files found.")
        return manifest

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
                chunk_id = f"{f.path.as_posix()}::p{page.page_index}::c{c.chunk_index}"
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

        if for_upsert_ids:
            # True sync: delete any previous vectors for this file before upserting new ones.
            store.delete_by_filter(
                filter_metadata={
                    "source_path": str(f.path),
                    "modality": "pdf",
                    "file_sha256": sha,
                }
            )

            store.upsert_embeddings(
                ids=for_upsert_ids,
                embeddings=for_upsert_embeddings,
                metadatas=for_upsert_metadatas,
            )

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
                "file_sha256": sha,
            }
        )

        embedding = embed_client.embed_image(str(f.path))
        vec_id = f"{f.path.as_posix()}::img0"

        store.upsert_embeddings(
            ids=[vec_id],
            embeddings=[embedding],
            metadatas=[
                {
                    "source_path": str(f.path),
                    "file_name": f.path.name,
                    "modality": "png",
                    "image_id": "img0",
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

    store_kwargs: Dict[str, Any] = {"api_key": settings.quadrant_api_key, "collection": args.collection or "data_multimodal"}
    if settings.quadrant_url:
        store_kwargs["url"] = settings.quadrant_url
    if settings.quadrant_path:
        store_kwargs["path"] = settings.quadrant_path

    store = QuadrantStore(**store_kwargs)

    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)

    embed_client = EmbedAnythingClient(model_name=settings.embedanything_model)

    manifest = ingest_changed_pdfs(
        data_dir=data_dir,
        settings=settings,
        store=store,
        embed_client=embed_client,
        manifest=manifest,
    )
    manifest = ingest_changed_pngs(
        data_dir=data_dir,
        settings=settings,
        store=store,
        embed_client=embed_client,
        manifest=manifest,
    )

    _save_manifest(manifest_path, manifest)
    print("Sync complete.")


if __name__ == "__main__":
    main()
