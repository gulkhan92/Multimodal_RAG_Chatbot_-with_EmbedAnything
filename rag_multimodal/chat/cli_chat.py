from __future__ import annotations

import argparse

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
    store = QuadrantStore(url=settings.quadrant_url, api_key=settings.quadrant_api_key)

    embed_client = EmbedAnythingClient(model_name=settings.embedanything_model)
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

        resp = answer_question(
            question=q,
            settings=settings,
            embed_client=embed_client,
            store=store,
            gemini=gemini,
            top_k=args.top_k,
        )

        print("\nAnswer:\n" + resp.answer)
        if resp.citations:
            print("\nCitations: " + ", ".join(f"[{i}]" for i in resp.citations))


if __name__ == "__main__":
    main()
