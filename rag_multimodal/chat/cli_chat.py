from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from rag_multimodal.chat.__init__ import __all__ as _unused  # keeps package import happy
from rag_multimodal.ingest.embed_anything import EmbedAnythingClient
from rag_multimodal.ingest.quadrant_store import QuadrantStore
from rag_multimodal.rag.answer import answer_question
from rag_multimodal.rag.gemini_client import GeminiClient
from rag_multimodal.settings import Settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    settings = Settings.from_env()
    store_kwargs = {"api_key": settings.quadrant_api_key}
    if settings.quadrant_url:
        store_kwargs["url"] = settings.quadrant_url
    if settings.quadrant_path:
        store_kwargs["path"] = settings.quadrant_path

    store = QuadrantStore(**store_kwargs)

    text_client = EmbedAnythingClient(model_name=settings.embedanything_text_model)
    image_client = EmbedAnythingClient(model_name=settings.embedanything_image_model)
    gemini = GeminiClient(api_key=settings.gemini_api_key)

    print("Multimodal RAG CLI. Type 'exit' to quit.")
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye.")
            return

        # Perform multimodal retrieval across separate collections
        # 1. Search PDFs (Text space)
        text_query_emb = text_client.embed_text([q])[0]
        store.collection = "data_pdf"
        text_chunks = store.similarity_search(embedding=text_query_emb, top_k=args.top_k)

        # 2. Search Images (CLIP space)
        image_query_emb = image_client.embed_text([q])[0]
        store.collection = "data_png"
        image_chunks = store.similarity_search(embedding=image_query_emb, top_k=args.top_k)

        # Combine results
        all_chunks = sorted(text_chunks + image_chunks, key=lambda x: x.score, reverse=True)[:args.top_k]

        # Note: You may need to update your `answer_question` signature in answer.py 
        # to accept the pre-retrieved chunks directly.
        from rag_multimodal.rag.prompt import build_gemini_prompt
        prompt = build_gemini_prompt(question=q, chunks=all_chunks)
        
        # Build multimodal contents list
        contents = [prompt]
        # Track unique images to avoid passing the same image multiple times
        seen_images = set()
        for ch in all_chunks:
            if ch.metadata.get("modality") == "png":
                img_path = ch.metadata.get("source_path")
                if img_path and img_path not in seen_images:
                    p = Path(img_path)
                    if p.exists():
                        contents.append(Image.open(p))
                        seen_images.add(img_path)

        gemini_resp = gemini.generate(contents=contents)
        answer_text = gemini_resp.text

        print("\nAnswer:\n" + answer_text)
        
        # Optional: Parse and display citations
        from rag_multimodal.rag.answer import parse_citations
        citations = parse_citations(answer_text)
        if citations:
            print("\nCitations: " + ", ".join(f"[{i}]" for i in citations))


if __name__ == "__main__":
    main()
