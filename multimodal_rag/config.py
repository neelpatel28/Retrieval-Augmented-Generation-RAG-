# ============================================================
#  config.py — Central configuration for Multimodal RAG
#  Edit this file to switch LLMs, models, and settings
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Backend ─────────────────────────────────────────────
# Options: "groq" | "ollama" | "openai"
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq")

# Groq settings (free tier — get key at console.groq.com)
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Ollama settings (fully local — install from ollama.com)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "mistral")

# OpenAI settings (paid)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Embedding Model ──────────────────────────────────────────
# Options: "bge-m3" | "clip" | "openai"
EMBED_MODEL     = os.getenv("EMBED_MODEL", "bge-m3")
EMBED_MODEL_ID  = "BAAI/bge-m3"           # HuggingFace model ID
CLIP_MODEL_ID   = "clip-ViT-B-32"         # For image embeddings
EMBED_DIMENSION = 1024                     # BGE-M3 output dim

# ── Vector Store ─────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION  = os.getenv("CHROMA_COLLECTION", "multimodal_rag")

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))

# ── Retrieval ────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", 6))

# ── Whisper (Video transcription) ───────────────────────────
# Options: "tiny" | "base" | "small" | "medium" | "large"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
VIDEO_FRAME_INTERVAL = int(os.getenv("VIDEO_FRAME_INTERVAL", 30))  # sample every N frames

# ── Output ───────────────────────────────────────────────────
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
