import os
from typing import List, Dict

import requests

from app.llm.base_client import BaseLLMClient


class OllamaClient(BaseLLMClient):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {"model": self.model, "messages": messages, "stream": False}
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
