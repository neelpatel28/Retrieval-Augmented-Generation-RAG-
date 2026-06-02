"""
RAG Pipeline - Core Module
===========================
Handles: PDF ingestion → chunking → embedding → FAISS vector store → retrieval
"""

import os
import re
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

import fitz  # PyMuPDF - best for preserving layout
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Chunk:
    """A single text chunk with metadata."""
    chunk_id: int
    text: str
    page_num: int
    source_file: str
    char_start: int
    char_end: int
    # For overlap tracking
    prev_overlap: str = ""
    next_overlap: str = ""

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "page_num": self.page_num,
            "source_file": self.source_file,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


# ─────────────────────────────────────────────
# STEP 1: PDF INGESTION
# ─────────────────────────────────────────────

class PDFIngestor:
    """
    Extracts text from PDFs using PyMuPDF (fitz).

    Why PyMuPDF?
    - Preserves reading order better than pdfplumber/PyPDF2
    - Handles multi-column layouts
    - Extracts text with block/line structure intact
    - Handles tables, headers, footers
    """

    def __init__(self, preserve_layout: bool = True):
        self.preserve_layout = preserve_layout

    def extract(self, pdf_path: str) -> List[Dict]:
        """
        Returns list of page dicts:
          { "page_num": int, "text": str, "block_count": int }
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"📄 Ingesting PDF: {pdf_path.name}")
        pages = []

        with fitz.open(str(pdf_path)) as doc:
            logger.info(f"   Total pages: {doc.page_count}")

            for page_num, page in enumerate(doc, start=1):
                if self.preserve_layout:
                    # get_text("blocks") returns text sorted by reading order
                    blocks = page.get_text("blocks", sort=True)
                    # Each block: (x0, y0, x1, y1, text, block_no, block_type)
                    # block_type 0 = text, 1 = image
                    text_blocks = [
                        b[4].strip()
                        for b in blocks
                        if b[6] == 0 and b[4].strip()  # only text blocks
                    ]
                    full_text = "\n".join(text_blocks)
                else:
                    full_text = page.get_text("text")

                # Clean up excessive whitespace while preserving paragraph breaks
                full_text = self._clean_text(full_text)

                if full_text:
                    pages.append({
                        "page_num": page_num,
                        "text": full_text,
                        "block_count": len(text_blocks) if self.preserve_layout else 0,
                    })
                    logger.info(f"   Page {page_num}: {len(full_text)} chars extracted")

        logger.info(f"✅ Extraction complete. {len(pages)} pages with text.")
        return pages

    def _clean_text(self, text: str) -> str:
        """Remove noise while preserving structure."""
        # Normalize multiple newlines to max 2 (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove page headers/footers patterns (lines with only numbers)
        lines = text.split('\n')
        lines = [l for l in lines if not re.match(r'^\s*\d+\s*$', l)]
        text = '\n'.join(lines)
        # Remove hyphenation across lines: "computa-\ntion" → "computation"
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        return text.strip()


# ─────────────────────────────────────────────
# STEP 2: TEXT CHUNKING
# ─────────────────────────────────────────────

class TextChunker:
    """
    Recursive character-based chunker with overlap.

    Strategy:
    1. Try to split on paragraph breaks (\n\n)
    2. Fall back to sentence boundaries (. ! ?)
    3. Fall back to word boundaries
    4. Hard split as last resort

    Overlap: each chunk shares N chars with adjacent chunks
    so context isn't lost at boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 800,       # characters per chunk (tune based on embedding model)
        chunk_overlap: int = 150,    # characters of overlap between chunks
        min_chunk_size: int = 100,   # discard chunks smaller than this
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Split hierarchy: paragraph → sentence → word
        self.separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

    def chunk_pages(self, pages: List[Dict], source_file: str) -> List[Chunk]:
        """Convert page-level text into overlapping chunks."""
        all_chunks = []
        chunk_id = 0

        for page in pages:
            page_chunks = self._split_text(page["text"])

            for raw_text in page_chunks:
                raw_text = raw_text.strip()
                if len(raw_text) < self.min_chunk_size:
                    continue

                chunk = Chunk(
                    chunk_id=chunk_id,
                    text=raw_text,
                    page_num=page["page_num"],
                    source_file=source_file,
                    char_start=0,   # simplified; track if needed
                    char_end=len(raw_text),
                )
                all_chunks.append(chunk)
                chunk_id += 1

        logger.info(f"✅ Chunking complete: {len(all_chunks)} chunks created")
        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """Recursively split text using separator hierarchy."""
        chunks = []
        self._recursive_split(text, self.separators, chunks)
        return chunks

    def _recursive_split(self, text: str, separators: List[str], result: List[str]):
        """Core recursive splitting with overlap injection."""
        if len(text) <= self.chunk_size:
            if text.strip():
                result.append(text)
            return

        separator = ""
        remaining_separators = []

        # Find the best separator to use
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                remaining_separators = separators[i + 1:]
                break

        # Split by chosen separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)  # character-level fallback

        current_chunk = ""

        for split in splits:
            if not split:
                continue

            candidate = current_chunk + (separator if current_chunk else "") + split

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                # Save current chunk if non-empty
                if current_chunk.strip():
                    if len(current_chunk) > self.chunk_size and remaining_separators:
                        # Too big, recurse with finer separator
                        self._recursive_split(current_chunk, remaining_separators, result)
                    else:
                        result.append(current_chunk)

                # Start new chunk with overlap from previous
                if current_chunk and self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + (separator if overlap_text else "") + split
                else:
                    current_chunk = split

        # Don't forget the last chunk
        if current_chunk.strip():
            result.append(current_chunk)


# ─────────────────────────────────────────────
# STEP 3: EMBEDDING GENERATION
# ─────────────────────────────────────────────

class EmbeddingEngine:
    """
    Generates dense vector embeddings using sentence-transformers.

    Model choice: 'all-MiniLM-L6-v2'
    - 384 dimensions, fast, good quality
    - Free, runs locally, no API key needed

    For production: consider 'BAAI/bge-large-en-v1.5' or OpenAI embeddings
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"🧠 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"   Embedding dimension: {self.dimension}")

    def embed_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> np.ndarray:
        """
        Embed all chunks. Returns numpy array of shape (N, dimension).
        Batching prevents OOM on large documents.
        """
        texts = [chunk.text for chunk in chunks]
        logger.info(f"🔢 Embedding {len(texts)} chunks in batches of {batch_size}...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # L2-normalize for cosine similarity
            convert_to_numpy=True,
        )

        logger.info(f"✅ Embeddings shape: {embeddings.shape}")
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)


# ─────────────────────────────────────────────
# STEP 4: VECTOR STORE (FAISS)
# ─────────────────────────────────────────────

class FAISSVectorStore:
    """
    FAISS-based vector store for fast similarity search.

    Index type: IndexFlatIP (Inner Product = cosine sim when normalized)
    - Exact search, no approximation
    - Good for < 100k chunks
    - For larger: use IndexIVFFlat or IndexHNSWFlat
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        # Inner product index (cosine similarity after L2 normalization)
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Add chunks and their embeddings to the index."""
        assert len(chunks) == embeddings.shape[0], "Mismatch between chunks and embeddings"
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info(f"✅ Vector store: {self.index.ntotal} vectors indexed")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Search for top-k most similar chunks.
        Returns list of (Chunk, score) tuples sorted by relevance.
        """
        scores, indices = self.index.search(query_embedding, top_k)
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 for empty slots
                continue
            results.append((self.chunks[idx], float(score)))

        return results

    def save(self, path: str):
        """Persist index and metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(path / "meta.json", "w") as f:
            json.dump({"dimension": self.dimension, "total": self.index.ntotal}, f)

        logger.info(f"💾 Vector store saved to: {path}")

    @classmethod
    def load(cls, path: str) -> "FAISSVectorStore":
        """Load a previously saved vector store."""
        path = Path(path)

        with open(path / "meta.json") as f:
            meta = json.load(f)

        store = cls(dimension=meta["dimension"])
        store.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)

        logger.info(f"📂 Loaded vector store: {store.index.ntotal} vectors")
        return store


# ─────────────────────────────────────────────
# STEP 5: RETRIEVAL WITH MMR
# ─────────────────────────────────────────────

class Retriever:
    """
    Retrieves relevant chunks for a query.

    Features:
    - Basic top-k similarity search
    - MMR (Maximal Marginal Relevance): balances relevance + diversity
      to avoid returning near-duplicate chunks
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_engine: EmbeddingEngine,
        top_k: int = 5,
        use_mmr: bool = True,
        mmr_lambda: float = 0.6,   # 1.0 = pure relevance, 0.0 = pure diversity
    ):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.top_k = top_k
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

    def retrieve(self, query: str) -> List[Tuple[Chunk, float]]:
        """Main retrieval method."""
        query_embedding = self.embedding_engine.embed_query(query)

        if self.use_mmr:
            return self._mmr_retrieve(query_embedding)
        else:
            return self.vector_store.search(query_embedding, self.top_k)

    def _mmr_retrieve(self, query_embedding: np.ndarray) -> List[Tuple[Chunk, float]]:
        """
        Maximal Marginal Relevance retrieval.
        1. Fetch 3x candidates
        2. Iteratively pick chunks that maximize: λ·relevance - (1-λ)·redundancy
        """
        # Fetch more candidates than needed
        candidates = self.vector_store.search(query_embedding, self.top_k * 3)

        if not candidates:
            return []

        selected = []
        candidate_texts = [c[0].text for c in candidates]
        candidate_embeddings = self.embedding_engine.model.encode(
            candidate_texts, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        remaining_indices = list(range(len(candidates)))

        for _ in range(min(self.top_k, len(candidates))):
            if not remaining_indices:
                break

            best_idx = None
            best_score = float("-inf")

            for i in remaining_indices:
                # Relevance: cosine similarity with query
                relevance = float(np.dot(query_embedding[0], candidate_embeddings[i]))

                # Redundancy: max similarity with already-selected chunks
                if selected:
                    selected_embeddings = np.array([candidate_embeddings[s] for s in selected])
                    redundancy = float(np.max(selected_embeddings @ candidate_embeddings[i]))
                else:
                    redundancy = 0.0

                mmr_score = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * redundancy

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(best_idx)
            remaining_indices.remove(best_idx)

        return [(candidates[i][0], candidates[i][1]) for i in selected]


# ─────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end orchestrator. Call once to build the index,
    then use .query() for all subsequent questions.
    """

    STORE_PATH = "vector_store"

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 5,
    ):
        self.embedding_engine = EmbeddingEngine(embedding_model)
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.ingestor = PDFIngestor()
        self.vector_store: Optional[FAISSVectorStore] = None
        self.retriever: Optional[Retriever] = None
        self.top_k = top_k
        self.indexed_file: Optional[str] = None

    def ingest_pdf(self, pdf_path: str, force_reindex: bool = False):
        """Full ingestion pipeline: PDF → chunks → embeddings → FAISS."""
        store_path = Path(self.STORE_PATH)

        # Load from cache if available
        if store_path.exists() and not force_reindex:
            logger.info("📂 Found existing index. Loading from cache...")
            self.vector_store = FAISSVectorStore.load(self.STORE_PATH)
            self.indexed_file = pdf_path
            self._init_retriever()
            return

        # Full pipeline
        pages = self.ingestor.extract(pdf_path)
        chunks = self.chunker.chunk_pages(pages, source_file=Path(pdf_path).name)
        embeddings = self.embedding_engine.embed_chunks(chunks)

        self.vector_store = FAISSVectorStore(dimension=self.embedding_engine.dimension)
        self.vector_store.add(chunks, embeddings)
        self.vector_store.save(self.STORE_PATH)

        self.indexed_file = pdf_path
        self._init_retriever()
        logger.info("🎉 RAG pipeline ready!")

    def _init_retriever(self):
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_engine=self.embedding_engine,
            top_k=self.top_k,
            use_mmr=True,
        )

    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve relevant chunks for a query."""
        if not self.retriever:
            raise RuntimeError("Pipeline not initialized. Call ingest_pdf() first.")

        results = self.retriever.retrieve(query)
        return [
            {**chunk.to_dict(), "score": round(score, 4)}
            for chunk, score in results
        ]
