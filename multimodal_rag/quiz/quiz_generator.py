# ============================================================
#  quiz/quiz_generator.py — MCQ quiz generation
# ============================================================

import json
import logging
import re
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError

from config import TOP_K

logger = logging.getLogger(__name__)


# ── Pydantic Schemas ─────────────────────────────────────────
class MCQOption(BaseModel):
    label:  str = Field(..., description="Option label: A, B, C, or D")
    text:   str = Field(..., description="Option text")


class MCQuestion(BaseModel):
    question:    str           = Field(..., description="The question text")
    options:     List[MCQOption] = Field(..., description="Exactly 4 options")
    answer:      str           = Field(..., description="Correct label: A, B, C, or D")
    explanation: str           = Field(..., description="Why the answer is correct")
    source:      str           = Field(default="", description="Source document")


class QuizOutput(BaseModel):
    topic:     str             = Field(..., description="Topic of the quiz")
    questions: List[MCQuestion]


# ── Quiz Generation ──────────────────────────────────────────
def generate_quiz(
    topic: str,
    n_questions: int = 5,
    difficulty: str = "medium",
    filter_type: Optional[str] = None,
) -> QuizOutput:
    """
    Generate MCQ quiz on a topic using retrieved context.

    Args:
        topic:       Topic to quiz on
        n_questions: Number of questions (1-15)
        difficulty:  "easy" | "medium" | "hard"
        filter_type: Optional doc type filter

    Returns:
        QuizOutput with validated MCQ questions
    """
    from vectorstore.chroma_store import query as chroma_query
    from utils.llm_provider import get_llm_client

    n_questions = max(1, min(n_questions, 15))

    # ── Retrieve context ──
    filter_meta = {"type": filter_type} if filter_type else None
    hits = chroma_query(topic, top_k=TOP_K + 2, filter_metadata=filter_meta)

    if not hits:
        raise ValueError("No relevant documents found for this topic. Please upload documents first.")

    context = "\n\n---\n\n".join([
        f"[Source: {h['metadata'].get('source', 'unknown')}]\n{h['text']}"
        for h in hits
    ])

    # ── Build prompt ──
    system_prompt = """You are an expert quiz creator. Generate MCQ questions strictly based on the provided context.
Return ONLY valid JSON — no markdown, no backticks, no explanation outside the JSON."""

    user_prompt = f"""Generate exactly {n_questions} multiple-choice questions on the topic: "{topic}"
Difficulty level: {difficulty}

Rules:
- Each question must have exactly 4 options labeled A, B, C, D
- Only ONE correct answer per question
- Include a clear explanation for the correct answer
- Base ALL questions on the context below — do not use outside knowledge
- Include the source document name in the "source" field

Return ONLY this JSON structure (no markdown, no extra text):
{{
  "topic": "{topic}",
  "questions": [
    {{
      "question": "...",
      "options": [
        {{"label": "A", "text": "..."}},
        {{"label": "B", "text": "..."}},
        {{"label": "C", "text": "..."}},
        {{"label": "D", "text": "..."}}
      ],
      "answer": "A",
      "explanation": "...",
      "source": "filename.pdf"
    }}
  ]
}}

Context:
{context}
"""

    # ── Call LLM ──
    llm = get_llm_client()
    raw = llm.chat(user_prompt, system_prompt=system_prompt)

    # ── Parse & validate ──
    return _parse_quiz_response(raw, topic)


def _parse_quiz_response(raw: str, topic: str) -> QuizOutput:
    """Parse and validate LLM JSON response into QuizOutput."""
    # Strip markdown fences if present
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Find JSON object
    match = re.search(r'\{.*\}', clean, re.DOTALL)
    if not match:
        raise ValueError(f"LLM did not return valid JSON.\nRaw response:\n{raw[:500]}")

    try:
        data = json.loads(match.group())
        return QuizOutput(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"[QUIZ] Validation failed: {e}\nRaw: {raw[:500]}")
        raise ValueError(f"Failed to parse quiz output: {e}")
