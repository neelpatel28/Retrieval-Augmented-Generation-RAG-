# ============================================================
#  utils/chunker.py — Split raw parsed chunks into smaller pieces
# ============================================================

import logging
from typing import List, Dict, Any
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def chunk_documents(raw_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split raw parsed chunks (pages/slides/segments) into
    smaller overlapping chunks suitable for embedding.

    Video and PPT chunks are kept as-is (natural boundaries).
    PDF and text chunks are split by token count.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError("Run: pip install langchain-text-splitters")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 4,   # ~4 chars per token estimate
        chunk_overlap=CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    result = []

    for chunk in raw_chunks:
        doc_type = chunk["metadata"].get("type", "text")

        # Natural boundaries — keep as-is
        if doc_type in ("pptx", "video_transcript", "video_frame"):
            result.append(chunk)
            continue

        # Split long text/pdf pages
        text = chunk["text"]
        if len(text) <= CHUNK_SIZE * 4:
            result.append(chunk)
            continue

        sub_texts = splitter.split_text(text)
        for i, sub in enumerate(sub_texts):
            new_chunk = {
                "text": sub,
                "metadata": {
                    **chunk["metadata"],
                    "sub_chunk": i,
                }
            }
            result.append(new_chunk)

    logger.info(f"[CHUNK] {len(raw_chunks)} raw → {len(result)} final chunks")
    return result
