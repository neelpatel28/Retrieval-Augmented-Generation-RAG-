# ============================================================
#  ingestion/pdf_parser.py — PDF ingestion using pymupdf
# ============================================================

import os
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a PDF file and return a list of chunks with metadata.
    Extracts: text per page + table text + image OCR (if pytesseract available)

    Returns:
        List of dicts: { text, metadata: { source, page, type, total_pages } }
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError("Run: pip install pymupdf")

    chunks = []
    file_name = Path(file_path).name

    logger.info(f"[PDF] Parsing: {file_name}")
    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc, start=1):
        # ── Extract text ──
        text = page.get_text("text").strip()

        # ── Extract tables as text (if pdfplumber available) ──
        table_text = _extract_tables(file_path, page_num)
        if table_text:
            text += "\n\n[TABLE]\n" + table_text

        # ── Extract images and OCR ──
        image_texts = _extract_image_text(page, doc)
        if image_texts:
            text += "\n\n[IMAGE TEXT]\n" + "\n".join(image_texts)

        if text.strip():
            chunks.append({
                "text": text,
                "metadata": {
                    "source":      file_name,
                    "file_path":   file_path,
                    "page":        page_num,
                    "total_pages": len(doc),
                    "type":        "pdf",
                }
            })

    doc.close()
    logger.info(f"[PDF] Extracted {len(chunks)} pages from {file_name}")
    return chunks


def _extract_tables(file_path: str, page_num: int) -> str:
    """Extract tables using pdfplumber (optional)."""
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            page = pdf.pages[page_num - 1]
            tables = page.extract_tables()
            rows = []
            for table in tables:
                for row in table:
                    cleaned = [str(cell).strip() if cell else "" for cell in row]
                    rows.append(" | ".join(cleaned))
            return "\n".join(rows)
    except Exception:
        return ""


def _extract_image_text(page, doc) -> List[str]:
    """OCR images inside PDF pages using pytesseract (optional)."""
    texts = []
    try:
        import pytesseract
        from PIL import Image
        import io

        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(image).strip()
            if ocr_text:
                texts.append(ocr_text)
    except Exception:
        pass  # OCR is optional
    return texts
