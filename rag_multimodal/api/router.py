from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends
from PIL import Image

from rag_multimodal.api.schemas import ChatRequest, ChatResponse
from rag_multimodal.auth_utils import RoleChecker
from rag_multimodal.ingest.embed_anything import EmbedAnythingClient
from rag_multimodal.ingest.quadrant_store import QuadrantStore
from rag_multimodal.rag.gemini_client import GeminiClient
from rag_multimodal.rag.prompt import build_gemini_prompt

router = APIRouter()


def build_chat_routes(
    *,
    text_client: EmbedAnythingClient,
    image_client: EmbedAnythingClient,
    gemini: GeminiClient,
    store: QuadrantStore,
) -> APIRouter:
    @router.post("/chat", response_model=ChatResponse, dependencies=[Depends(RoleChecker(["viewer"]))])
    async def chat(request: ChatRequest) -> ChatResponse:
        q = request.question

        text_query_emb = text_client.embed_text([q])[0]
        store.collection = "data_pdf"
        text_chunks = store.similarity_search(embedding=text_query_emb, top_k=request.top_k)

        image_query_emb = image_client.embed_text([q])[0]
        store.collection = "data_png"
        image_chunks = store.similarity_search(embedding=image_query_emb, top_k=request.top_k)

        all_chunks = sorted(text_chunks + image_chunks, key=lambda x: x.score, reverse=True)[: request.top_k]

        prompt = build_gemini_prompt(question=q, chunks=all_chunks)

        contents = [prompt]
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

        return ChatResponse(
            answer=gemini_resp.text.strip(),
            sources=[ch.metadata.get("source_path") for ch in all_chunks],
        )

    return router
