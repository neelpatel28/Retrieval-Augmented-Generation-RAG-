# ============================================================
#  ingestion/text_parser.py — Plain text file ingestion
# ============================================================

import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json", ".html"}


def parse_text(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a plain text file. Returns the full content as one chunk
    (chunking into smaller pieces happens in the pipeline layer).

    Returns:
        List with single dict: { text, metadata: { source, type } }
    """
    file_name = Path(file_path).name
    ext = Path(file_path).suffix.lower()

    logger.info(f"[TEXT] Parsing: {file_name}")

    encodings = ["utf-8", "latin-1", "cp1252"]
    text = None

    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        logger.warning(f"[TEXT] Could not decode {file_name}, skipping.")
        return []

    if not text.strip():
        return []

    # For JSON: pretty-print key-value pairs for better embedding
    if ext == ".json":
        import json
        try:
            data = json.loads(text)
            text = json.dumps(data, indent=2)
        except Exception:
            pass

    # For CSV: convert to readable row format
    if ext == ".csv":
        text = _csv_to_text(file_path)

    return [{
        "text": text,
        "metadata": {
            "source":    file_name,
            "file_path": file_path,
            "type":      "text",
            "extension": ext,
        }
    }]


def _csv_to_text(file_path: str) -> str:
    """Convert CSV rows into human-readable text for embedding."""
    import csv
    rows = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            row_text = ", ".join(f"{k}: {v}" for k, v in row.items())
            rows.append(f"Row {i+1}: {row_text}")
    return "\n".join(rows)
