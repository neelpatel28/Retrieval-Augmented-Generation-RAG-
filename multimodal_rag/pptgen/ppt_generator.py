# ============================================================
#  pptgen/ppt_generator.py — AI-powered PPT generation
# ============================================================

import json
import logging
import os
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from config import TOP_K, OUTPUT_DIR

logger = logging.getLogger(__name__)


# ── Pydantic Schemas ─────────────────────────────────────────
class Slide(BaseModel):
    title:         str        = Field(..., description="Slide title")
    bullets:       List[str]  = Field(..., description="3-5 bullet points")
    speaker_notes: str        = Field(default="", description="Speaker notes")
    layout:        str        = Field(default="content", description="title|content|two_col")


class PresentationPlan(BaseModel):
    title:    str         = Field(..., description="Presentation title")
    subtitle: str         = Field(default="", description="Subtitle")
    slides:   List[Slide]


# ── PPT Generation ───────────────────────────────────────────
def generate_ppt(
    topic: str,
    n_slides: int = 8,
    style: str = "academic",
    filter_type: Optional[str] = None,
) -> str:
    """
    Generate a .pptx presentation on a topic from retrieved context.

    Args:
        topic:       Topic for the presentation
        n_slides:    Target number of slides (3-15)
        style:       "academic" | "professional" | "minimal"
        filter_type: Optional doc type filter

    Returns:
        Path to the generated .pptx file
    """
    from vectorstore.chroma_store import query as chroma_query
    from utils.llm_provider import get_llm_client

    n_slides = max(3, min(n_slides, 15))

    # ── Retrieve context ──
    filter_meta = {"type": filter_type} if filter_type else None
    hits = chroma_query(topic, top_k=TOP_K + 4, filter_metadata=filter_meta)

    if not hits:
        raise ValueError("No relevant documents found. Please upload documents first.")

    context = "\n\n---\n\n".join([
        f"[Source: {h['metadata'].get('source', 'unknown')}]\n{h['text']}"
        for h in hits
    ])

    # ── Generate slide plan via LLM ──
    plan = _generate_slide_plan(topic, n_slides, style, context)

    # ── Render with python-pptx ──
    output_path = _render_pptx(plan, style)
    logger.info(f"[PPT] Generated: {output_path}")
    return output_path


def _generate_slide_plan(topic, n_slides, style, context) -> PresentationPlan:
    """Ask LLM to create a structured slide plan."""
    from utils.llm_provider import get_llm_client

    system_prompt = "You are a professional presentation designer. Return ONLY valid JSON — no markdown, no backticks."

    user_prompt = f"""Create a {n_slides}-slide presentation on: "{topic}"
Style: {style}

Slide structure:
- Slide 1: Title slide
- Slides 2 to {n_slides-1}: Content slides (3-5 bullet points each)
- Slide {n_slides}: Summary / Conclusion

Rules:
- Each bullet point: concise, max 12 words
- Speaker notes: 1-2 sentences expanding on the slide
- Base content ONLY on the provided context

Return ONLY this JSON:
{{
  "title": "...",
  "subtitle": "Generated from uploaded documents",
  "slides": [
    {{
      "title": "...",
      "bullets": ["...", "...", "..."],
      "speaker_notes": "...",
      "layout": "title"
    }}
  ]
}}

Context:
{context}
"""

    llm = get_llm_client()
    raw = llm.chat(user_prompt, system_prompt=system_prompt)

    # Parse
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match = re.search(r'\{.*\}', clean, re.DOTALL)
    if not match:
        raise ValueError(f"LLM did not return valid JSON for slide plan.\n{raw[:500]}")

    data = json.loads(match.group())
    return PresentationPlan(**data)


def _render_pptx(plan: PresentationPlan, style: str) -> str:
    """Render the slide plan into an actual .pptx file."""
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt, Emu
        from pptx.dml.color import RGBColor
        from pptx.enum.text import PP_ALIGN
    except ImportError:
        raise ImportError("Run: pip install python-pptx")

    # Style presets
    styles = {
        "academic":     {"bg": (15, 40, 77),   "title": (255,255,255), "body": (220,230,240), "accent": (41,182,246)},
        "professional": {"bg": (255,255,255),  "title": (30, 30, 90),  "body": (50, 50, 50),  "accent": (0,120,215)},
        "minimal":      {"bg": (250,250,250),  "title": (20, 20, 20),  "body": (60, 60, 60),  "accent": (100,100,100)},
    }
    palette = styles.get(style, styles["academic"])

    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)

    for i, slide_data in enumerate(plan.slides):
        is_title_slide = (i == 0) or (slide_data.layout == "title")
        layout_idx = 0 if is_title_slide else 1
        slide_layout = prs.slide_layouts[layout_idx]
        slide = prs.slides.add_slide(slide_layout)

        # ── Background ──
        bg = slide.background.fill
        bg.solid()
        bg.fore_color.rgb = RGBColor(*palette["bg"])

        # ── Title ──
        title_shape = slide.shapes.title
        if title_shape:
            title_shape.text = slide_data.title
            tf = title_shape.text_frame
            for para in tf.paragraphs:
                for run in para.runs:
                    run.font.color.rgb = RGBColor(*palette["title"])
                    run.font.size = Pt(36 if is_title_slide else 28)
                    run.font.bold = True

        # ── Content / Bullets ──
        if not is_title_slide and slide_data.bullets:
            content_placeholder = None
            for ph in slide.placeholders:
                if ph.placeholder_format.idx == 1:
                    content_placeholder = ph
                    break

            if content_placeholder:
                tf = content_placeholder.text_frame
                tf.clear()
                tf.word_wrap = True

                for j, bullet in enumerate(slide_data.bullets):
                    if j == 0:
                        p = tf.paragraphs[0]
                    else:
                        p = tf.add_paragraph()
                    p.text  = f"• {bullet}"
                    p.level = 0
                    for run in p.runs:
                        run.font.size  = Pt(18)
                        run.font.color.rgb = RGBColor(*palette["body"])

        # ── Subtitle (title slide only) ──
        if is_title_slide and i == 0:
            for ph in slide.placeholders:
                if ph.placeholder_format.idx == 1:
                    ph.text = plan.subtitle
                    for para in ph.text_frame.paragraphs:
                        for run in para.runs:
                            run.font.color.rgb = RGBColor(*palette["body"])
                            run.font.size = Pt(20)
                    break

        # ── Speaker notes ──
        if slide_data.speaker_notes:
            notes_slide = slide.notes_slide
            notes_slide.notes_text_frame.text = slide_data.speaker_notes

    # ── Footer with source attribution ──
    _add_footer(prs, plan.title)

    # ── Save ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = re.sub(r'[^\w\s-]', '', plan.title)[:40].strip().replace(' ', '_')
    output_path = os.path.join(OUTPUT_DIR, f"{safe_topic}_{timestamp}.pptx")
    prs.save(output_path)
    return output_path


def _add_footer(prs, title: str):
    """Add a subtle footer to all slides."""
    from pptx.util import Inches, Pt, Emu
    from pptx.dml.color import RGBColor

    for slide in prs.slides:
        txBox = slide.shapes.add_textbox(
            Inches(0.2), Inches(7.1), Inches(6), Inches(0.3)
        )
        tf = txBox.text_frame
        tf.text = f"Generated by Multimodal RAG  |  {title}"
        for para in tf.paragraphs:
            for run in para.runs:
                run.font.size  = Pt(9)
                run.font.color.rgb = RGBColor(150, 160, 170)
