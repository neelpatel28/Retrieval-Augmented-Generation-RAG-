# ============================================================
#  utils/llm_provider.py — Dynamic LLM backend switcher
# ============================================================

import logging
from config import (
    LLM_BACKEND, GROQ_API_KEY, GROQ_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL
)

logger = logging.getLogger(__name__)


def get_llm_client():
    """
    Returns a unified LLM client based on LLM_BACKEND in config.
    All clients expose a .chat(messages, system) interface via wrapper.
    """
    if LLM_BACKEND == "groq":
        return GroqClient()
    elif LLM_BACKEND == "ollama":
        return OllamaClient()
    elif LLM_BACKEND == "openai":
        return OpenAIClient()
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND}. Choose: groq | ollama | openai")


# ── Groq Client ──────────────────────────────────────────────
class GroqClient:
    def __init__(self):
        try:
            from groq import Groq
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set in .env")
            self.client = Groq(api_key=GROQ_API_KEY)
            self.model  = GROQ_MODEL
            logger.info(f"[LLM] Using Groq — model: {self.model}")
        except ImportError:
            raise ImportError("Run: pip install groq")

    def chat(self, user_message: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=4096,
        )
        return response.choices[0].message.content


# ── Ollama Client ────────────────────────────────────────────
class OllamaClient:
    def __init__(self):
        try:
            import requests
            self.base_url = OLLAMA_BASE_URL
            self.model    = OLLAMA_MODEL
            # Quick health check
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            r.raise_for_status()
            logger.info(f"[LLM] Using Ollama — model: {self.model} at {self.base_url}")
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
                f"Install from https://ollama.com and run: ollama pull {OLLAMA_MODEL}\nError: {e}"
            )

    def chat(self, user_message: str, system_prompt: str = "") -> str:
        import requests, json
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [],
        }
        if system_prompt:
            payload["messages"].append({"role": "system", "content": system_prompt})
        payload["messages"].append({"role": "user", "content": user_message})

        r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]


# ── OpenAI Client ─────────────────────────────────────────────
class OpenAIClient:
    def __init__(self):
        try:
            from openai import OpenAI
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set in .env")
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.model  = OPENAI_MODEL
            logger.info(f"[LLM] Using OpenAI — model: {self.model}")
        except ImportError:
            raise ImportError("Run: pip install openai")

    def chat(self, user_message: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=4096,
        )
        return response.choices[0].message.content
