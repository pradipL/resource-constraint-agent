import os
from typing import List, Dict

from app.llm.base_client import BaseLLMClient

_LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))


class OpenAIClient(BaseLLMClient):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.max_completion_tokens = int(os.getenv("OPENAI_MAX_COMPLETION_TOKENS", "16384"))

    def chat(self, messages: List[Dict[str, str]], max_completion_tokens: int | None = None) -> str:
        from openai import APIConnectionError, APITimeoutError, OpenAI

        kwargs: dict = {"api_key": self.api_key, "timeout": _LLM_TIMEOUT}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        try:
            client = OpenAI(**kwargs)
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_completion_tokens or self.max_completion_tokens,
            )
        except APITimeoutError as exc:
            raise RuntimeError(f"OpenAI call timed out after {_LLM_TIMEOUT}s.") from exc
        except APIConnectionError as exc:
            raise RuntimeError(f"OpenAI connection failed: {exc}") from exc

        return resp.choices[0].message.content

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
