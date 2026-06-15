# ============================================================
#  ingestion/pipeline.py — Master ingestion router
#  Routes files to the correct parser by extension
# ============================================================

import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Extension → parser mapping
EXTENSION_MAP = {
    ".pdf":  "pdf",
    ".pptx": "ppt",
    ".ppt":  "ppt",
    ".txt":  "text",
    ".md":   "text",
    ".rst":  "text",
    ".csv":  "text",
    ".json": "text",
    ".html": "text",
    ".htm":  "text",
    ".mp4":  "video",
    ".avi":  "video",
    ".mov":  "video",
    ".mkv":  "video",
    ".webm": "video",
}


def ingest_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Route a single file to the correct parser.
    Returns list of raw chunks (not yet embedded).
    """
    ext = Path(file_path).suffix.lower()
    parser_type = EXTENSION_MAP.get(ext)

    if parser_type is None:
        logger.warning(f"[INGEST] Unsupported file type: {ext} — skipping {file_path}")
        return []

    logger.info(f"[INGEST] Processing {Path(file_path).name} as [{parser_type}]")

    if parser_type == "pdf":
        from ingestion.pdf_parser import parse_pdf
        return parse_pdf(file_path)

    elif parser_type == "ppt":
        from ingestion.ppt_parser import parse_ppt
        return parse_ppt(file_path)

    elif parser_type == "text":
        from ingestion.text_parser import parse_text
        return parse_text(file_path)

    elif parser_type == "video":
        from ingestion.video_parser import parse_video
        return parse_video(file_path)

    return []


def ingest_directory(dir_path: str) -> List[Dict[str, Any]]:
    """
    Ingest all supported files in a directory recursively.
    Returns combined list of all chunks.
    """
    all_chunks = []
    dir_path = Path(dir_path)

    supported_files = [
        f for f in dir_path.rglob("*")
        if f.is_file() and f.suffix.lower() in EXTENSION_MAP
    ]

    logger.info(f"[INGEST] Found {len(supported_files)} files in {dir_path}")

    for file_path in supported_files:
        try:
            chunks = ingest_file(str(file_path))
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"[INGEST] Failed on {file_path.name}: {e}")

    logger.info(f"[INGEST] Total raw chunks extracted: {len(all_chunks)}")
    return all_chunks
