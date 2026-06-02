# 📚 RAG Document Chatbot — Complete Guide

A production-quality Retrieval-Augmented Generation (RAG) system that lets you
chat with any PDF document. Built for interview demonstration with clean,
well-commented code covering every stage of the RAG pipeline.

---

## 🏗 Architecture Overview

```
                    PDF File
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: PDF INGESTION (rag_pipeline.py → PDFIngestor)      │
│  • PyMuPDF extracts text block-by-block, preserving order   │
│  • Handles multi-column layouts, cleans hyphenation         │
│  Output: List of {page_num, text} dicts                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: TEXT CHUNKING (rag_pipeline.py → TextChunker)      │
│  • Recursive splitter: tries \n\n → \n → sentence → word    │
│  • Chunk size: 800 chars, overlap: 150 chars                │
│  • Overlap prevents context loss at chunk boundaries        │
│  Output: List[Chunk] with page metadata                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: EMBEDDING (rag_pipeline.py → EmbeddingEngine)      │
│  • Model: all-MiniLM-L6-v2 (384 dims, runs locally)         │
│  • L2-normalized → cosine similarity via dot product        │
│  • Batched encoding (64 chunks/batch)                       │
│  Output: numpy array (N × 384)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 4: VECTOR STORE (rag_pipeline.py → FAISSVectorStore)  │
│  • IndexFlatIP = exact inner product search                 │
│  • Saved to disk for fast reload (no re-embedding)          │
│  Output: Searchable FAISS index + chunk metadata            │
└──────────────────────┬──────────────────────────────────────┘
                       │ (query time)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: RETRIEVAL (rag_pipeline.py → Retriever)            │
│  • MMR: Maximal Marginal Relevance                          │
│  • λ=0.6: balances relevance vs. diversity                  │
│  • Returns top-5 chunks (configurable)                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: LLM GENERATION (llm_client.py → LLMClient)         │
│  • Prompt: system instructions + context chunks + query     │
│  • Model: Claude Sonnet via Anthropic API                   │
│  • Fallback: extractive answer if no API key                │
│  Output: Grounded natural language answer                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 7: INTERFACE                                          │
│  • Web UI: Flask + beautiful HTML/JS chatbot (server.py)    │
│  • CLI: terminal chatbot (cli_chat.py)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Setup Instructions

### 1. Create and activate a virtual environment

```bash
python -m venv rag_env

# On Linux/Mac:
source rag_env/bin/activate

# On Windows:
rag_env\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> First run will download the embedding model (~90 MB). Cached afterwards.

### 3. Set your Anthropic API key (optional but recommended)

```bash
# Linux/Mac
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows
set ANTHROPIC_API_KEY=sk-ant-...
```

Without the key, the system uses an **extractive fallback** — it returns
the most relevant chunk verbatim. Still useful for demos.

---

## 💻 Running the System

### Option A: Web UI (recommended)

```bash
python server.py
```

Open `http://localhost:5000` in your browser.

1. Click or drag-and-drop a PDF in the sidebar
2. Wait for indexing (progress bar)
3. Type your question and press Enter

### Option B: CLI

```bash
python cli_chat.py path/to/your/document.pdf
```

Optional flags:
```bash
python cli_chat.py doc.pdf --chunk-size 600 --overlap 100 --top-k 7
python cli_chat.py doc.pdf --force-reindex   # re-index even if cache exists
```

---

## 📁 Project Structure

```
rag_system/
├── rag_pipeline.py     # Core: ingestion, chunking, embedding, FAISS
├── llm_client.py       # LLM integration + conversation management
├── server.py           # Flask web server + REST API
├── cli_chat.py         # Terminal interface
├── requirements.txt    # Python dependencies
├── static/
│   └── index.html      # Web chatbot UI (single file)
└── vector_store/       # Auto-created: FAISS index + chunk cache
    ├── index.faiss
    ├── chunks.pkl
    └── meta.json
```

---

## 🧠 Key Design Decisions (Interview Talking Points)

### Why PyMuPDF over PyPDF2/pdfplumber?
- `get_text("blocks", sort=True)` returns blocks in visual reading order
- Handles multi-column academic papers correctly
- Built-in hyphenation fix across line breaks

### Why character-level chunking over sentence splitting?
- Language-agnostic (works for any language)
- Deterministic and fast (no NLP model needed)
- Recursive fallback prevents over-large chunks

### Why FAISS IndexFlatIP?
- **Exact search** — no approximation errors
- Inner product = cosine similarity (after L2 normalization)
- Scales to ~500k vectors before needing IVF/HNSW

### Why MMR over plain top-k?
- Top-k can return near-identical chunks (repeated paragraphs, headers)
- MMR's λ parameter lets you tune relevance vs. diversity trade-off
- Prevents context window waste on redundant content

### Why sentence-transformers locally?
- Zero API cost for embedding
- No rate limits during indexing
- `all-MiniLM-L6-v2` is battle-tested, runs on CPU in seconds

---

## 🔧 Improving the System (Interview Suggestions)

| Area | Current | Upgrade |
|------|---------|---------|
| Embedding model | all-MiniLM-L6-v2 (384d) | BAAI/bge-large-en-v1.5 (1024d) or OpenAI text-embedding-3-small |
| Vector index | FAISS FlatIP | FAISS HNSW (faster at scale) or Chroma (persistent) |
| Chunking | Character-based | Semantic chunking (split on topic shifts) |
| Retrieval | MMR | Hybrid search: BM25 + semantic (reciprocal rank fusion) |
| Reranking | None | Add Cohere Rerank or cross-encoder reranker |
| Hallucination control | System prompt | Add faithfulness scoring (NLI model) |
| Tables/figures | Skipped | Use Camelot for tables, LLaVA for figures |
| Multi-doc | Single PDF | Document store with metadata filtering |
| Eval | None | RAGAS framework for retrieval + generation quality |

---

## 🎯 Example Queries to Demonstrate

- "What is the main contribution of this paper?"
- "Summarize the methodology section"
- "What datasets were used in the experiments?"
- "What are the limitations mentioned by the authors?"
- "What future work do the authors suggest?"
