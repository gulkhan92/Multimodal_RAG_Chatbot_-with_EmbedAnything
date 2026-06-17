from __future__ import annotations

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm

from rag_multimodal.api.schemas import ChatRequest, ChatResponse, TokenResponse
from rag_multimodal.auth_utils import RoleChecker, create_access_token
from rag_multimodal.ingest.embed_anything import EmbedAnythingClient
from rag_multimodal.ingest.quadrant_store import QuadrantStore
from rag_multimodal.rag.gemini_client import GeminiClient
from rag_multimodal.rag.prompt import build_gemini_prompt
from rag_multimodal.settings import Settings

from rag_multimodal.api.router import router as base_router
from rag_multimodal.api.router import build_chat_routes

app = FastAPI(title="Multimodal RAG API (refactored)")

settings = Settings.from_env()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients (keep same core behavior as api_main.py)
text_client = EmbedAnythingClient(model_name=settings.embedanything_text_model)
image_client = EmbedAnythingClient(model_name=settings.embedanything_image_model)
gemini = GeminiClient(api_key=settings.gemini_api_key)
store = QuadrantStore(
    path=settings.quadrant_path,
    url=settings.quadrant_url,
    api_key=settings.quadrant_api_key,
)

# ---- Routes ----

# Build chat routes (typed schemas + pydantic validation)
chat_router = build_chat_routes(
    text_client=text_client,
    image_client=image_client,
    gemini=gemini,
    store=store,
)
app.include_router(chat_router)


@app.post("/token", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Mock verification for MVP (keep behavior identical to api_main.py)
    if form_data.username not in ["admin_user", "staff_user"]:
        # api_main.py used 400 Incorrect username or password
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Keep core logic identical to api_main.py
    q = request.question

    text_query_emb = text_client.embed_text([q])[0]
    store.collection = "data_pdf"
    text_chunks = store.similarity_search(embedding=text_query_emb, top_k=request.top_k)

    image_query_emb = image_client.embed_text([q])[0]
    store.collection = "data_png"
    image_chunks = store.similarity_search(embedding=image_query_emb, top_k=request.top_k)

    all_chunks = sorted(text_chunks + image_chunks, key=lambda x: x.score, reverse=True)[: request.top_k]

    prompt = build_gemini_prompt(question=q, chunks=all_chunks)

    from PIL import Image
    from pathlib import Path

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

    return {
        "answer": gemini_resp.text,
        "sources": [ch.metadata.get("source_path") for ch in all_chunks],
    }


from rag_multimodal.auth_utils import RoleChecker


@app.post("/ingest/sync")
async def sync_data(_=Depends(RoleChecker(["admin"]))):
    # Placeholder for triggering sync_vector_db logic (keep behavior identical)
    return {"message": "Sync triggered successfully"}
