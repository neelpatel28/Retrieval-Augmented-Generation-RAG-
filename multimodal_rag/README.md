# 🧠 Multimodal RAG System

A complete, production-ready RAG system supporting **Text, PDF, PPT, and Video** with:
- 💬 **Q&A** — Ask questions grounded in your documents
- 📝 **Quiz Generator** — Auto-generate MCQ quizzes on any topic
- 📊 **PPT Generator** — Create AI-powered presentations from your content

---

## ⚡ Quick Start

### 1. Clone / extract the project
```bash
cd multimodal_rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Video support** also needs ffmpeg:
> - Ubuntu/Debian: `sudo apt install ffmpeg`
> - Mac: `brew install ffmpeg`
> - Windows: Download from https://ffmpeg.org/download.html

> **OCR support** (for images in PDFs) needs Tesseract:
> - Ubuntu: `sudo apt install tesseract-ocr`
> - Mac: `brew install tesseract`
> - Windows: https://github.com/UB-Mannheim/tesseract/wiki

### 3. Set up your API key
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get a free key at: https://console.groq.com
```

### 4. Index your documents
```bash
# Index a whole folder
python index.py --input ./data/uploads

# Index a single file
python index.py --file ./data/uploads/lecture.pdf

# Reset index and re-index
python index.py --input ./data/uploads --reset
```

### 5. Launch the UI
```bash
streamlit run ui/app.py
```
Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
multimodal_rag/
├── config.py                  # All settings — edit here
├── .env.example               # Copy to .env and fill keys
├── index.py                   # CLI indexing tool
├── requirements.txt
│
├── ingestion/
│   ├── pipeline.py            # Master router (routes files to parsers)
│   ├── pdf_parser.py          # PDF → text + tables + OCR
│   ├── ppt_parser.py          # PPTX → slides + notes
│   ├── text_parser.py         # TXT / MD / CSV / JSON
│   └── video_parser.py        # Video → Whisper transcript + frame OCR
│
├── embeddings/
│   └── embedder.py            # BGE-M3 (default) or OpenAI embeddings
│
├── vectorstore/
│   └── chroma_store.py        # ChromaDB wrapper (add, query, stats)
│
├── retrieval/
│   └── rag_engine.py          # Retrieve + LLM answer generation
│
├── quiz/
│   └── quiz_generator.py      # MCQ quiz with Pydantic validation
│
├── pptgen/
│   └── ppt_generator.py       # AI slide plan → python-pptx rendering
│
├── utils/
│   ├── llm_provider.py        # Dynamic LLM switcher (Groq/Ollama/OpenAI)
│   └── chunker.py             # Text splitter
│
├── ui/
│   └── app.py                 # Streamlit web interface
│
├── data/uploads/              # Put your files here
└── outputs/                   # Generated PPTs saved here
```

---

## 🔧 Switching LLM Backends

Edit `.env`:

```bash
# Use Groq (free, fast — recommended)
LLM_BACKEND=groq
GROQ_API_KEY=your_key_here

# Use Ollama (fully local, no API key)
LLM_BACKEND=ollama
OLLAMA_MODEL=mistral   # Run: ollama pull mistral

# Use OpenAI (paid)
LLM_BACKEND=openai
OPENAI_API_KEY=your_key_here
```

---

## 🎯 Supported File Types

| Type | Extensions | Parser |
|------|-----------|--------|
| PDF | `.pdf` | pymupdf + pdfplumber |
| PowerPoint | `.pptx`, `.ppt` | python-pptx |
| Text | `.txt`, `.md`, `.rst` | Built-in |
| Data | `.csv`, `.json` | Built-in |
| Video | `.mp4`, `.avi`, `.mov`, `.mkv` | Whisper + OpenCV |

---

## 💡 Tips

- **First run** downloads BGE-M3 model (~1GB) — takes a few minutes
- **Video files** are slow to index — Whisper transcription takes ~10% of video duration
- **For GPU speedup**: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- **ChromaDB** stores data in `./chroma_db/` — safe to delete to start fresh

---

## 📦 Dependencies

All free and open-source:
- **pymupdf** — PDF parsing
- **python-pptx** — PPT parsing + generation
- **openai-whisper** — Video transcription (runs locally)
- **sentence-transformers** — BGE-M3 embeddings (runs locally)
- **chromadb** — Vector store (runs locally)
- **groq** — Free LLM API
- **streamlit** — Web UI
