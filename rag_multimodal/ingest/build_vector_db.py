from __future__ import annotations

import argparse
from pathlib import Path

from rag_multimodal.ingest.run_ingest import ingest_pngs, ingest_pdfs, _default_collection_for_modality
from rag_multimodal.ingest.quadrant_store import QuadrantStore
from rag_multimodal.ingest.embed_anything import EmbedAnythingClient
from rag_multimodal.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build/update the Quadrant vector DB from every supported file in the data folder."
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Folder containing input documents.")
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Optional Quadrant collection name. If omitted, a default is used.",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    data_dir = Path(args.data_dir)

    store = QuadrantStore(url=settings.quadrant_url, api_key=settings.quadrant_api_key)
    store.collection = args.collection or _default_collection_for_modality("all")

    # Ingest functions will (a) create embeddings and (b) upsert into Quadrant.
    # EmbedAnythingClient construction is handled inside run_ingest ingest functions.
    ingest_pdfs(data_dir=data_dir, settings=settings, store=store)
    ingest_pngs(data_dir=data_dir, settings=settings, store=store)

    print(f"Vector DB build/update complete for data dir: {data_dir}")


if __name__ == "__main__":
    main()
