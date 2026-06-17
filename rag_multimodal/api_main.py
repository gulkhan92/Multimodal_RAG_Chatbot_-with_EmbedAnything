from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
import io

from rag_multimodal.settings import Settings
from rag_multimodal.auth_utils import create_access_token, get_current_user, RoleChecker, User
from rag_multimodal.ingest.embed_anything import EmbedAnythingClient
from rag_multimodal.ingest.quadrant_store import QuadrantStore
from rag_multimodal.rag.gemini_client import GeminiClient
from rag_multimodal.rag.prompt import build_gemini_prompt

app = FastAPI(title="Multimodal RAG API")
settings = Settings.from_env()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients
text_client = EmbedAnythingClient(model_name=settings.embedanything_text_model)
image_client = EmbedAnythingClient(model_name=settings.embedanything_image_model)
gemini = GeminiClient(api_key=settings.gemini_api_key)
store = QuadrantStore(
    path=settings.quadrant_path, 
    url=settings.quadrant_url, 
    api_key=settings.quadrant_api_key
)

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Mock verification for MVP
    if form_data.username not in ["admin_user", "staff_user"]:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/chat", dependencies=[Depends(RoleChecker(["viewer"]))])
async def chat(request: ChatRequest):
    q = request.question
    
    # Multimodal Retrieval
    text_query_emb = text_client.embed_text([q])[0]
    store.collection = "data_pdf"
    text_chunks = store.similarity_search(embedding=text_query_emb, top_k=request.top_k)

    image_query_emb = image_client.embed_text([q])[0]
    store.collection = "data_png"
    image_chunks = store.similarity_search(embedding=image_query_emb, top_k=request.top_k)

    all_chunks = sorted(text_chunks + image_chunks, key=lambda x: x.score, reverse=True)[:request.top_k]
    
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
    
    return {
        "answer": gemini_resp.text,
        "sources": [ch.metadata.get("source_path") for ch in all_chunks]
    }

@app.post("/ingest/sync", dependencies=[Depends(RoleChecker(["admin"]))])
async def sync_data():
    # Placeholder for triggering sync_vector_db logic
    return {"message": "Sync triggered successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)