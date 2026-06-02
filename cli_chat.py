"""
CLI Chatbot Interface
======================
Run this directly to chat via terminal:
  python cli_chat.py path/to/document.pdf
"""

import sys
import time
import logging
import argparse
from pathlib import Path

from rag_pipeline import RAGPipeline
from llm_client import LLMClient, ConversationManager

logging.basicConfig(level=logging.WARNING)  # quiet in CLI mode

BANNER = """
╔══════════════════════════════════════════════════════╗
║          📚  RAG Document Chatbot  🤖                ║
║          Type 'quit' to exit | 'clear' to reset      ║
╚══════════════════════════════════════════════════════╝
"""

def main():
    parser = argparse.ArgumentParser(description="RAG CLI Chatbot")
    parser.add_argument("pdf", help="Path to the PDF document")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force-reindex", action="store_true",
                        help="Re-index even if cache exists")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    print(BANNER)

    # Build pipeline
    print(f"⚙️  Initializing RAG pipeline...")
    pipeline = RAGPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        top_k=args.top_k,
    )

    print(f"📄 Ingesting: {pdf_path.name}")
    t0 = time.time()
    pipeline.ingest_pdf(str(pdf_path), force_reindex=args.force_reindex)
    print(f"✅ Indexed {pipeline.vector_store.index.ntotal} chunks in {time.time()-t0:.1f}s\n")

    llm = LLMClient()
    conversation = ConversationManager()

    print(f"💬 Ready! Ask anything about: {pdf_path.name}")
    print("─" * 55)

    while True:
        try:
            query = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if query.lower() == "clear":
            conversation.clear()
            print("🗑️  Conversation cleared.")
            continue

        # Retrieve + answer
        print("🔍 Retrieving...", end=" ", flush=True)
        t0 = time.time()
        chunks = pipeline.retrieve(query)
        print(f"({len(chunks)} chunks, {int((time.time()-t0)*1000)}ms)")

        print("🤖 Generating answer...", end=" ", flush=True)
        t0 = time.time()
        result = llm.answer(query, chunks)
        print(f"({int((time.time()-t0)*1000)}ms)")

        print("\n" + "─" * 55)
        print(f"🤖 Answer ({result['model']}):\n")
        print(result["answer"])

        print("\n📎 Sources:")
        for i, chunk in enumerate(chunks[:3], 1):
            preview = chunk["text"][:80].replace("\n", " ")
            print(f"   {i}. Page {chunk['page_num']} (score={chunk['score']:.3f}): {preview}...")

        print("─" * 55)

        conversation.add("user", query)
        conversation.add("assistant", result["answer"])


if __name__ == "__main__":
    main()
