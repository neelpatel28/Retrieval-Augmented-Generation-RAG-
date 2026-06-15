# ============================================================
#  ingestion/ppt_parser.py — PPT/PPTX ingestion
# ============================================================

import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def parse_ppt(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse a .pptx file. Extracts text + speaker notes per slide.

    Returns:
        List of dicts: { text, metadata: { source, slide, type, total_slides } }
    """
    try:
        from pptx import Presentation
    except ImportError:
        raise ImportError("Run: pip install python-pptx")

    chunks = []
    file_name = Path(file_path).name

    logger.info(f"[PPT] Parsing: {file_name}")
    prs = Presentation(file_path)

    for slide_num, slide in enumerate(prs.slides, start=1):
        texts = []

        # ── Slide title ──
        if slide.shapes.title and slide.shapes.title.text.strip():
            texts.append(f"[TITLE] {slide.shapes.title.text.strip()}")

        # ── All text shapes ──
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = para.text.strip()
                    if line:
                        texts.append(line)

        # ── Speaker notes ──
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                texts.append(f"[SPEAKER NOTES] {notes}")

        full_text = "\n".join(texts)
        if full_text.strip():
            chunks.append({
                "text": full_text,
                "metadata": {
                    "source":       file_name,
                    "file_path":    file_path,
                    "slide":        slide_num,
                    "total_slides": len(prs.slides),
                    "type":         "pptx",
                }
            })

    logger.info(f"[PPT] Extracted {len(chunks)} slides from {file_name}")
    return chunks
