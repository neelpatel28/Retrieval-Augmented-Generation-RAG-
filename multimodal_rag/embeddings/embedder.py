# ============================================================
#  embeddings/embedder.py — Text embedding using BGE-M3
# ============================================================

import logging
from typing import List
from config import EMBED_MODEL, EMBED_MODEL_ID

logger = logging.getLogger(__name__)

_model_instance = None  # singleton


def get_embedder():
    """Return singleton embedder instance based on EMBED_MODEL config."""
    global _model_instance
    if _model_instance is None:
        if EMBED_MODEL == "bge-m3":
            _model_instance = BGEM3Embedder()
        elif EMBED_MODEL == "openai":
            _model_instance = OpenAIEmbedder()
        else:
            raise ValueError(f"Unknown EMBED_MODEL: {EMBED_MODEL}. Choose: bge-m3 | openai")
    return _model_instance


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings. Returns list of embedding vectors."""
    return get_embedder().embed(texts)


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    return get_embedder().embed([query])[0]


# ── BGE-M3 (local, free) ─────────────────────────────────────
class BGEM3Embedder:
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[EMBED] Loading BGE-M3 model: {EMBED_MODEL_ID}")
            self.model = SentenceTransformer(EMBED_MODEL_ID)
            logger.info("[EMBED] BGE-M3 model loaded successfully")
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

    def embed(self, texts: List[str]) -> List[List[float]]:
        # BGE-M3 benefits from instruction prefix for retrieval
        prefixed = [f"Represent this sentence: {t}" for t in texts]
        embeddings = self.model.encode(prefixed, normalize_embeddings=True)
        return embeddings.tolist()


# ── OpenAI Embeddings (paid) ─────────────────────────────────
class OpenAIEmbedder:
    def __init__(self):
        try:
            from openai import OpenAI
            from config import OPENAI_API_KEY
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model  = "text-embedding-3-small"
            logger.info(f"[EMBED] Using OpenAI embeddings: {self.model}")
        except ImportError:
            raise ImportError("Run: pip install openai")

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]
