# Multimodal RAG Chatbot (EmbedAnything + Gemini + Quadrant)

This repo builds a **multi-source Retrieval-Augmented Generation (RAG) chatbot** that can answer questions over everything under `data/`.

## Initial MVP scope (implemented)
- Ingest: **PDF** + **PNG** from `data/`
- Embeddings: **EmbedAnything**
- Vector DB: **Quadrant**
- LLM answering: **Gemini**
- Interface: **CLI chatbot** (first version)

## Quick start

### 1) Install dependencies
This project uses **Poetry** (via `pyproject.toml`).

```bash
poetry install
poetry shell
```

### 2) Configure environment
```bash
cp .env.example .env
```

Set:
- `GEMINI_API_KEY`
- Quadrant connection settings (as required by your Quadrant setup)
- EmbedAnything model/params (if required)

### 3) Ingest local data into Quadrant
```bash
python -m rag_multimodal.ingest.run_ingest --data-dir data
```

### 4) Ask questions (CLI chatbot)
```bash
python -m rag_multimodal.chat.cli_chat --data-dir data
```

## Project layout
- `rag_multimodal/ingest/*`: ingestion pipeline (PDF + PNG -> chunks -> embeddings -> Quadrant upsert)
- `rag_multimodal/rag/*`: retrieval + prompt construction for Gemini
- `rag_multimodal/chat/*`: CLI chat loop

## Notes
This repo currently contains only the `data/` folder. All code is scaffolded from scratch for the MVP.
If any of the EmbedAnything / Quadrant import paths differ from what’s assumed, update the small wrapper modules in:
- `rag_multimodal/ingest/embed_anything.py`
- `rag_multimodal/ingest/quadrant_store.py`
