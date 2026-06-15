# ============================================================
#  index.py — Ingest files and build the vector index
#  Usage: python index.py --input ./data/uploads
#         python index.py --file ./data/uploads/lecture.pdf
#         python index.py --reset  (clears the index)
# ============================================================

import argparse
import logging
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LOG_LEVEL
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("index")


def run_indexing(input_path: str = None, file_path: str = None, reset: bool = False):
    from vectorstore.chroma_store import delete_collection, get_stats, add_chunks
    from ingestion.pipeline import ingest_file, ingest_directory
    from utils.chunker import chunk_documents

    if reset:
        logger.info("Resetting vector store...")
        delete_collection()
        logger.info("Vector store cleared.")
        if not input_path and not file_path:
            return

    # ── Ingest ──
    if file_path:
        logger.info(f"Ingesting single file: {file_path}")
        raw_chunks = ingest_file(file_path)
    elif input_path:
        logger.info(f"Ingesting directory: {input_path}")
        raw_chunks = ingest_directory(input_path)
    else:
        logger.error("Provide --input <dir> or --file <path>")
        return

    if not raw_chunks:
        logger.warning("No content extracted. Check file types and parsers.")
        return

    # ── Chunk ──
    logger.info(f"Splitting {len(raw_chunks)} raw chunks...")
    chunks = chunk_documents(raw_chunks)

    # ── Embed + Store ──
    logger.info(f"Embedding and storing {len(chunks)} chunks...")
    n = add_chunks(chunks)

    # ── Stats ──
    stats = get_stats()
    logger.info(f"✅ Indexing complete!")
    logger.info(f"   Total chunks in store : {stats['total_chunks']}")
    logger.info(f"   Breakdown by type     : {stats['by_type']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store")
    parser.add_argument("--input",  type=str, help="Directory of files to ingest")
    parser.add_argument("--file",   type=str, help="Single file to ingest")
    parser.add_argument("--reset",  action="store_true", help="Clear index before ingesting")
    args = parser.parse_args()

    run_indexing(
        input_path=args.input,
        file_path=args.file,
        reset=args.reset,
    )
