from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag_multimodal.api.logging_config import setup_logging
from rag_multimodal.ingest.embed_anything import EmbedAnythingClient
from rag_multimodal.ingest.quadrant_store import QuadrantStore
from rag_multimodal.rag.gemini_client import GeminiClient
from rag_multimodal.settings import Settings


# Import the route handlers from app.py
from rag_multimodal.api.app import (
    manager,
    login,
    signup,
    sync_data,
    websocket_endpoint,
)
from rag_multimodal.api.router import build_chat_routes

# --- Application Setup ---

app = FastAPI(title="Multimodal RAG API")

# 1. Load settings first to ensure all environment variables are available.
settings = Settings.from_env()

# 2. Configure logging.
setup_logging(log_manager=manager)

# 3. Initialize clients with the loaded settings.
text_client = EmbedAnythingClient(model_name=settings.embedanything_text_model)
image_client = EmbedAnythingClient(model_name=settings.embedanything_image_model)
gemini = GeminiClient(api_key=settings.gemini_api_key)
store = QuadrantStore(
    path=settings.quadrant_path,
    url=settings.quadrant_url,
    api_key=settings.quadrant_api_key,
)

# 4. Add CORS middleware to allow the frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Build and include routers.
chat_router = build_chat_routes(text_client=text_client, image_client=image_client, gemini=gemini, store=store)
app.include_router(chat_router)
app.add_api_route("/token", login, methods=["POST"])
app.add_api_route("/users/signup", signup, methods=["POST"], status_code=201)
app.add_api_route("/ingest/sync", sync_data, methods=["POST"])
app.add_api_websocket_route("/ws/logs", websocket_endpoint)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
