from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Simulated cost rate for local models (e.g. Ollama) that have no real API cost.
# Formula: cost = tokens_used * (0.01 / 1000)
_SIMULATED_RATE = 0.01 / 1000

# Price per 1 million tokens (input_usd, output_usd) for hosted providers.
# Models NOT listed here fall back to _SIMULATED_RATE applied to total tokens.
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o":          (5.00,  15.00),
    "gpt-4o-mini":     (0.15,   0.60),
    # Anthropic
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-haiku":    (0.25,  1.25),
}


@dataclass
class CallRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost_usd(self) -> float:
        pricing = MODEL_PRICING.get(self.model) or next(
            (v for k, v in MODEL_PRICING.items() if self.model.startswith(k)), None
        )
        if pricing:
            input_price, output_price = pricing
            return (
                self.prompt_tokens * input_price
                + self.completion_tokens * output_price
            ) / 1_000_000
        # Simulated cost for local/Ollama models
        return self.total_tokens * _SIMULATED_RATE


@dataclass
class CostTracker:
    records: List[CallRecord] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.records.append(CallRecord(model, prompt_tokens, completion_tokens))

    # ------------------------------------------------------------------ #
    # Aggregates                                                           #
    # ------------------------------------------------------------------ #

    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        calls = len(self.records)
        if not calls:
            return "No LLM calls recorded."

        lines = [
            "── Token & Cost Summary ─────────────────────",
            f"  LLM calls         : {calls}",
            f"  Prompt tokens     : {self.total_prompt_tokens():,}",
            f"  Completion tokens : {self.total_completion_tokens():,}",
            f"  Total tokens      : {self.total_tokens():,}",
            f"  Estimated cost    : ${self.total_cost_usd():.6f}",
            "─────────────────────────────────────────────",
        ]
        return "\n".join(lines)
