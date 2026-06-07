from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str = ""


class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, input: str) -> ToolResult:
        pass

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
