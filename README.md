# Enterprise Multimodal RAG Chatbot

The Enterprise Multimodal RAG Chatbot is a state-of-the-art information synthesis engine designed for high-fidelity cross-modal intelligence. By harmonizing disparate data streams—including unstructured documents, complex PDFs, and visual assets—the platform delivers strategically grounded insights powered by the latest advancements in Retrieval-Augmented Generation. Leveraging a decentralized vector architecture and Google Gemini's generative capabilities, it transforms siloed organizational data into a unified, actionable knowledge base with enterprise-grade precision and scalability.

## 🚀 Key Features
- **Dual-Model Embeddings**: Uses **BERT** for precise text representation and **CLIP** for visual content understanding.
- **Vector Search**: High-speed retrieval using **Qdrant** with modality-specific collections.
- **Grounded Reasoning**: Multimodal context (text and images) is passed to **Gemini 2.5 Flash** to ensure accurate, context-aware answers.
- **Enterprise UI/UX**: A professional React-based dashboard with session management and secure authentication.
- **Incremental Sync**: Intelligent data pipeline that only re-processes changed or new files.

## 🏗 Architecture Flow

```mermaid
graph TD
    subgraph Ingestion_Layer
        direction TB
        Data[Local Data Store /data] --> Dispatcher{File Dispatcher}
        
        subgraph Text_Pipeline
            Dispatcher -- Documents --> T_Ext[Text/PDF Extractors]
            T_Ext --> T_Chunk[Recursive Chunker]
            T_Chunk --> T_Embed[BERT Embedder: 384-dim]
            T_Embed --> V_Text[(Qdrant: data_text/pdf)]
        end
        
        subgraph Vision_Pipeline
            Dispatcher -- Images --> V_Load[Visual Asset Loader]
            V_Load --> V_Embed[CLIP Embedder: 512-dim]
            V_Embed --> V_Img[(Qdrant: data_png)]
        end
    end

    subgraph Retrieval_RAG
        direction TB
        UI[User Interface] --> API[FastAPI Orchestrator]
        API --> Query_Proc[Multi-Modal Query Embedding]
        
        subgraph Dual_Search
            Query_Proc -- 384-dim Vector --> V_Text
            Query_Proc -- 512-dim Vector --> V_Img
        end
        
        Dual_Search --> Aggregator[Context Aggregator & Re-ranker]
        Aggregator --> Gemini[Gemini 2.5 Flash]
        Gemini --> Response[Grounded Insights]
    end
```

## 🛠 Technologies
- **Language**: Python 3.11+
- **Frontend**: React, TypeScript, Vite
- **Backend**: FastAPI, Uvicorn
- **AI/ML**: EmbedAnything (BERT & CLIP), Google Generative AI (Gemini)
- **Database**: Qdrant (Local Persistent Mode)

## 📋 Prerequisites
- Python 3.11+
- Node.js & npm
- Google Gemini API Key

## ⚙️ Installation & Configuration

### 1. Backend Setup
This project uses **Poetry** for dependency management.

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
