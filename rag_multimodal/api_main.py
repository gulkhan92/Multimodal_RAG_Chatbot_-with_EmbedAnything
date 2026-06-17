"""
Backward-compatible entrypoint.

This module keeps the existing `rag_multimodal.api_main:app` import path working,
while the actual FastAPI app lives in `rag_multimodal.api.app`.
"""

from rag_multimodal.api.app import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
