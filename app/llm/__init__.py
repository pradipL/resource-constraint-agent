import os

from app.llm.base_client import BaseLLMClient
from app.llm.ollama_client import OllamaClient
from app.llm.openai_client import OpenAIClient


def get_llm_client() -> BaseLLMClient:
    """Factory — returns the client for the active LLM_PROVIDER (ollama | openai)."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "openai":
        return OpenAIClient()
    return OllamaClient()
