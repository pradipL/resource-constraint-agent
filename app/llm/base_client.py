from abc import ABC, abstractmethod
from typing import List, Dict


class BaseLLMClient(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], max_completion_tokens: int | None = None) -> str: ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...
