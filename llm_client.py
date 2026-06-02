"""
LLM Integration Module
========================
Connects retrieved context to Claude (via Anthropic API) to generate answers.
Also includes a local fallback using a simple extractive approach.
"""

import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

class PromptBuilder:
    """
    Constructs the RAG prompt sent to the LLM.

    Key design decisions:
    - Strict grounding: LLM must ONLY use provided context
    - Source attribution: encourages citing page numbers
    - Fallback instruction: say "I don't know" rather than hallucinate
    """

    SYSTEM_PROMPT = """You are a precise document assistant. Your job is to answer questions
based STRICTLY on the provided document excerpts.

Rules:
1. Only use information from the provided context.
2. If the answer is not in the context, say: "The document does not contain information about this."
3. Cite the page number(s) when possible, e.g. (Page 3).
4. Be concise but complete. Do not pad your answer.
5. If the question is ambiguous, answer the most likely interpretation.
"""

    def build(self, query: str, chunks: List[Dict]) -> str:
        """Build the full user prompt with context injected."""
        context_blocks = []

        for i, chunk in enumerate(chunks, 1):
            context_blocks.append(
                f"[Excerpt {i} | Page {chunk['page_num']} | Relevance: {chunk['score']:.3f}]\n"
                f"{chunk['text']}"
            )

        context_str = "\n\n---\n\n".join(context_blocks)

        return f"""Here are the relevant excerpts from the document:

{context_str}

---

Question: {query}

Answer based only on the excerpts above:"""


# ─────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────

class LLMClient:
    """
    Calls the Anthropic Claude API to generate answers.

    Falls back gracefully if API key is missing.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self.prompt_builder = PromptBuilder()
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize Anthropic client if API key available."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
                logger.info("✅ Anthropic client initialized")
            except ImportError:
                logger.warning("anthropic package not installed. Using fallback.")
        else:
            logger.warning("⚠️  ANTHROPIC_API_KEY not set. Using extractive fallback.")

    def answer(self, query: str, chunks: List[Dict]) -> Dict:
        """
        Generate an answer for the query given retrieved chunks.
        Returns dict with 'answer', 'model', 'chunks_used'.
        """
        if not chunks:
            return {
                "answer": "No relevant content found in the document for your query.",
                "model": "none",
                "chunks_used": 0,
            }

        if self._client:
            return self._claude_answer(query, chunks)
        else:
            return self._fallback_answer(query, chunks)

    def _claude_answer(self, query: str, chunks: List[Dict]) -> Dict:
        """Call Claude API."""
        prompt = self.prompt_builder.build(query, chunks)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=PromptBuilder.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            answer_text = response.content[0].text

        except Exception as e:
            logger.error(f"Claude API error: {e}")
            answer_text = f"[API Error: {str(e)}]\n\n" + self._extract_top_chunk(chunks)

        return {
            "answer": answer_text,
            "model": self.model,
            "chunks_used": len(chunks),
        }

    def _fallback_answer(self, query: str, chunks: List[Dict]) -> Dict:
        """
        Simple extractive fallback when no LLM is available.
        Returns the most relevant chunk as the 'answer'.
        """
        top_chunk = chunks[0]
        answer = (
            f"[Extractive Answer — No LLM configured]\n\n"
            f"Most relevant passage (Page {top_chunk['page_num']}, "
            f"relevance={top_chunk['score']:.3f}):\n\n"
            f"{top_chunk['text']}"
        )
        return {
            "answer": answer,
            "model": "extractive-fallback",
            "chunks_used": len(chunks),
        }

    def _extract_top_chunk(self, chunks: List[Dict]) -> str:
        if chunks:
            return f"Relevant content from page {chunks[0]['page_num']}:\n{chunks[0]['text']}"
        return ""


# ─────────────────────────────────────────────
# CONVERSATION MANAGER
# ─────────────────────────────────────────────

class ConversationManager:
    """
    Manages multi-turn chat history.
    Keeps last N turns to stay within context limits.
    """

    def __init__(self, max_history: int = 10):
        self.history: List[Dict] = []
        self.max_history = max_history

    def add(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to history."""
        entry = {"role": role, "content": content}
        if metadata:
            entry["metadata"] = metadata
        self.history.append(entry)

        # Trim old history
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

    def get_history(self) -> List[Dict]:
        return self.history

    def clear(self):
        self.history = []
        logger.info("🗑️  Conversation history cleared.")
