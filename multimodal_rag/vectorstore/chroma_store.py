# ============================================================
#  vectorstore/chroma_store.py — ChromaDB vector store
# ============================================================

import logging
import uuid
from typing import List, Dict, Any, Optional

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

_client = None
_collection = None


def get_collection():
    """Return (or create) the ChromaDB collection singleton."""
    global _client, _collection
    if _collection is None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("Run: pip install chromadb")

        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"[CHROMA] Collection '{CHROMA_COLLECTION}' ready. "
                    f"Documents: {_collection.count()}")
    return _collection


def add_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Embed and add chunks to ChromaDB.
    chunks: list of { text, metadata }
    Returns number of chunks added.
    """
    from embeddings.embedder import embed_texts

    if not chunks:
        return 0

    collection = get_collection()

    # Split into batches to avoid memory issues
    batch_size = 64
    total_added = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        texts     = [c["text"] for c in batch]
        metadatas = [_sanitize_metadata(c["metadata"]) for c in batch]
        ids       = [str(uuid.uuid4()) for _ in batch]

        logger.info(f"[CHROMA] Embedding batch {i//batch_size + 1} "
                    f"({len(batch)} chunks)...")
        embeddings = embed_texts(texts)

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total_added += len(batch)

    logger.info(f"[CHROMA] Added {total_added} chunks. "
                f"Total in store: {collection.count()}")
    return total_added


def query(
    query_text: str,
    top_k: int = 6,
    filter_metadata: Optional[Dict] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve top-K most relevant chunks for a query.
    Returns list of { text, metadata, distance }
    """
    from embeddings.embedder import embed_query

    collection = get_collection()
    if collection.count() == 0:
        logger.warning("[CHROMA] Collection is empty. Please ingest documents first.")
        return []

    query_embedding = embed_query(query_text)

    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if filter_metadata:
        kwargs["where"] = filter_metadata

    results = collection.query(**kwargs)

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        hits.append({
            "text":     doc,
            "metadata": meta,
            "score":    round(1 - dist, 4),  # cosine similarity
        })

    return hits


def get_stats() -> Dict[str, Any]:
    """Return collection statistics."""
    collection = get_collection()
    count = collection.count()

    # Count by type
    if count > 0:
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        type_counts = {}
        for m in all_meta:
            t = m.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
    else:
        type_counts = {}

    return {
        "total_chunks": count,
        "by_type": type_counts,
        "collection": CHROMA_COLLECTION,
        "persist_dir": CHROMA_PERSIST_DIR,
    }


def delete_collection():
    """Delete and reset the entire collection."""
    global _client, _collection
    if _client:
        _client.delete_collection(CHROMA_COLLECTION)
        _collection = None
        logger.info(f"[CHROMA] Collection '{CHROMA_COLLECTION}' deleted.")


def _sanitize_metadata(meta: Dict) -> Dict:
    """ChromaDB requires all metadata values to be str/int/float/bool."""
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean
