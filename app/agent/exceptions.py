class LLMLimitExceeded(Exception):
    """Raised when the maximum number of LLM calls for a run has been reached."""

    def __init__(self, limit: int):
        super().__init__(f"LLM call limit of {limit} reached.")
        self.limit = limit


class TokenLimitReached(Exception):
    """Raised when total tokens cross MAX_TOKEN_LIMIT, or when the provider rejects a call."""

    def __init__(self, current: int = 0, limit: int = 0):
        msg = (
            f"Total token limit of {limit:,} exceeded (current: {current:,})."
            if limit else "Token limit reached."
        )
        super().__init__(msg)
        self.current = current
        self.limit = limit


class BudgetExceeded(Exception):
    """Raised when accumulated LLM cost crosses the configured USD ceiling."""

    def __init__(self, current: float, ceiling: float):
        super().__init__(
            f"Cost budget of ${ceiling:.2f} exceeded (current: ${current:.6f})."
        )
        self.current = current
        self.ceiling = ceiling
