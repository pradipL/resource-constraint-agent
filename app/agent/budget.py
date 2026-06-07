from dataclasses import dataclass

from app.settings import settings


@dataclass
class Budget:
    max_iterations: int = settings.max_llm_calls
    max_tokens: int = 100_000
    used_iterations: int = 0
    used_tokens: int = 0

    def exhausted(self) -> bool:
        return (
            self.used_iterations >= self.max_iterations
            or self.used_tokens >= self.max_tokens
        )

    def tick(self, tokens: int = 0) -> None:
        self.used_iterations += 1
        self.used_tokens += tokens

    def summary(self) -> str:
        return (
            f"Iterations: {self.used_iterations}/{self.max_iterations} | "
            f"Tokens: {self.used_tokens}/{self.max_tokens}"
        )
