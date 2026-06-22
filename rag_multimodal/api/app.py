from __future__ import annotations

import logging
import asyncio
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from rag_multimodal.api.schemas import ChatRequest, ChatResponse, TokenResponse
from rag_multimodal.auth_utils import RoleChecker, create_access_token, verify_password, get_db, get_password_hash
from rag_multimodal.ingest.embed_anything import EmbedAnythingClient
from rag_multimodal.database import User as DBUser
from rag_multimodal.ingest.quadrant_store import QuadrantStore
from rag_multimodal.rag.gemini_client import GeminiClient
from rag_multimodal.rag.prompt import build_gemini_prompt
from rag_multimodal.settings import Settings

from rag_multimodal.api.logging_config import setup_logging
from rag_multimodal.api.router import router as base_router
from rag_multimodal.api.router import build_chat_routes

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

app = FastAPI(title="Multimodal RAG API (refactored)")
setup_logging(log_manager=manager)

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
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
chat_router = build_chat_routes(
    text_client=text_client,
    image_client=image_client,
    gemini=gemini,
    store=store,
)
app.include_router(chat_router)


@app.post("/token", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db=Depends(get_db)):
    from fastapi import HTTPException

    user = db.query(DBUser).filter(DBUser.username == form_data.username).first()
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/users/signup", status_code=status.HTTP_201_CREATED)
async def signup(user: UserCreate, db=Depends(get_db)):
    from rag_multimodal.database import Role
    db_user = db.query(DBUser).filter(DBUser.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    viewer_role = db.query(Role).filter(Role.name == "viewer").first()
    if not viewer_role:
        raise HTTPException(status_code=500, detail="Default 'viewer' role not found")

    hashed_password = get_password_hash(user.password)
    new_user = DBUser(username=user.username, email=user.email, hashed_password=hashed_password, roles=[viewer_role])
    db.add(new_user)
    db.commit()
    return {"message": "User created successfully"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # Keep core logic identical to api_main.py
    logging.info(f"Received chat request: question='{request.question}'")
    q = request.question

    logging.info("Embedding query for text search.")
    text_query_emb = text_client.embed_text([q])[0]
    store.collection = "data_pdf"
    logging.info("Searching PDF collection.")
    text_chunks = store.similarity_search(embedding=text_query_emb, top_k=request.top_k)
    
    # Search generic text files (txt, md, docx)
    store.collection = "data_text"
    logging.info("Searching text collection.")
    generic_text_chunks = store.similarity_search(embedding=text_query_emb, top_k=request.top_k)
    
    logging.info("Embedding query for image search.")
    image_query_emb = image_client.embed_text([q])[0]
    store.collection = "data_png"
    logging.info("Searching image collection.")
    image_chunks = store.similarity_search(embedding=image_query_emb, top_k=request.top_k)

    all_chunks = sorted(text_chunks + generic_text_chunks + image_chunks, key=lambda x: x.score, reverse=True)[: request.top_k]
    logging.info(f"Retrieved {len(all_chunks)} chunks for context.")

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

    logging.info(f"Sending {len(contents)} parts to Gemini.")
    gemini_resp = gemini.generate(contents=contents)
    logging.info("Received response from Gemini.")

    return {
        "answer": gemini_resp.text,
        "sources": [ch.metadata.get("source_path") for ch in all_chunks],
    }


from rag_multimodal.auth_utils import RoleChecker


@app.post("/ingest/sync")
async def sync_data(_=Depends(RoleChecker(["admin"]))):
    # Placeholder for triggering sync_vector_db logic (keep behavior identical)
    return {"message": "Sync triggered successfully"}


@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(60)  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)
