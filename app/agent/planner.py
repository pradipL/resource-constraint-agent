import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReActStep:
    thought: Optional[str]
    action: Optional[str]
    action_input: Optional[str]
    final_answer: Optional[str]

    @property
    def is_done(self) -> bool:
        return self.final_answer is not None

    @property
    def has_action(self) -> bool:
        return bool(self.action and self.action_input)


class ReActParser:
    """Parse a raw LLM response into a ReActStep."""

    _THOUGHT_RE = re.compile(
        r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL
    )
    _ACTION_RE = re.compile(r"Action:\s*(.+?)(?=\n|$)")
    _ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.+?)(?=\n|$)", re.DOTALL)
    _FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)

    def parse(self, text: str) -> ReActStep:
        return ReActStep(
            thought=self._get(self._THOUGHT_RE, text),
            action=self._get(self._ACTION_RE, text),
            action_input=self._get(self._ACTION_INPUT_RE, text),
            final_answer=self._get(self._FINAL_RE, text),
        )

    @staticmethod
    def _get(pattern: re.Pattern, text: str) -> Optional[str]:
        m = pattern.search(text)
        return m.group(1).strip() if m else None
