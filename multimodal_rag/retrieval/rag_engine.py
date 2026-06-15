# ============================================================
#  retrieval/rag_engine.py — RAG query answering engine
# ============================================================

import logging
from typing import List, Dict, Any, Optional
from config import TOP_K

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert AI assistant with access to a knowledge base.
Answer the user's question using ONLY the provided context below.
Rules:
- Be accurate, concise, and informative.
- Always cite your sources using the format: [Source: <filename>, page/slide/timestamp].
- If multiple sources support the answer, cite all of them.
- If the answer is NOT in the context, say: "I couldn't find relevant information in the uploaded documents."
- Do NOT make up information not present in the context.
"""


def answer_query(
    query: str,
    filter_type: Optional[str] = None,
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve → prompt → answer.

    Args:
        query:       User's question
        filter_type: Optional filter — 'pdf' | 'pptx' | 'video_transcript' | 'text'
        top_k:       Number of chunks to retrieve

    Returns:
        { answer, sources, context_used }
    """
    from vectorstore.chroma_store import query as chroma_query
    from utils.llm_provider import get_llm_client

    # ── Step 1: Retrieve ──
    filter_meta = {"type": filter_type} if filter_type else None
    hits = chroma_query(query, top_k=top_k, filter_metadata=filter_meta)

    if not hits:
        return {
            "answer": "No relevant documents found. Please upload and index some documents first.",
            "sources": [],
            "context_used": [],
        }

    # ── Step 2: Build context ──
    context_parts = []
    sources = []

    for i, hit in enumerate(hits):
        meta = hit["metadata"]
        source_label = _format_source(meta)
        context_parts.append(f"[{i+1}] {source_label}\n{hit['text']}")
        sources.append({
            "label":  source_label,
            "score":  hit["score"],
            "type":   meta.get("type", "unknown"),
            "source": meta.get("source", ""),
        })

    context = "\n\n---\n\n".join(context_parts)

    # ── Step 3: Build prompt ──
    user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    # ── Step 4: Call LLM ──
    llm = get_llm_client()
    answer = llm.chat(user_prompt, system_prompt=SYSTEM_PROMPT)

    return {
        "answer":       answer,
        "sources":      sources,
        "context_used": hits,
    }


def _format_source(meta: Dict) -> str:
    """Format a human-readable source citation from metadata."""
    source = meta.get("source", "unknown")
    doc_type = meta.get("type", "")

    if doc_type == "pdf":
        page = meta.get("page", "?")
        return f"{source} (Page {page})"
    elif doc_type == "pptx":
        slide = meta.get("slide", "?")
        return f"{source} (Slide {slide})"
    elif doc_type in ("video_transcript", "video_frame"):
        ts = meta.get("timestamp", "")
        seg = meta.get("segment", ts)
        return f"{source} (Video @ {seg})"
    else:
        return source
