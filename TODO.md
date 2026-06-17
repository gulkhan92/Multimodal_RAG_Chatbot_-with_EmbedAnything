# TODO - Multimodal RAG MVP

## Progress
- [x] Created project scaffolding files: `pyproject.toml`, `.env.example`, `README.md`
- [ ] Create Python package skeleton: `rag_multimodal/`
- [x] Implement ingestion pipeline for:
  - [x] PDF text extraction + chunking + embedding + Quadrant upsert (wiring pending EmbedAnything/Quadrant APIs)
  - [x] PNG embedding + Quadrant upsert (wiring pending EmbedAnything/Quadrant APIs)
- [x] Implement retrieval pipeline:
  - [x] query embedding (text) + Quadrant topK retrieval (Quadrant API wiring pending)
  - [ ] context assembly with citations metadata
- [x] Implement Gemini prompt + answer generation (grounded)
- [x] Implement CLI chatbot:
  - [x] `python -m rag_multimodal.chat.cli_chat --data-dir data`
- [x] Implement `python -m rag_multimodal.ingest.run_ingest --data-dir data` (ingest everything)
- [ ] Add basic tests / smoke checks (manual) + run instructions

## Follow-ups (after MVP works)
- [ ] Add OCR fallback for images if needed
- [ ] Add support for audio/video
- [ ] Add streaming responses in CLI / optional web UI
