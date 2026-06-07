from .state import AgentState, StepType


class Reflector:
    """Summarise an agent run after completion."""

    def reflect(self, state: AgentState) -> str:
        searches = [s for s in state.steps if s.type == StepType.ACTION]
        observations = [s for s in state.steps if s.type == StepType.OBSERVATION]

        lines = [
            "--- Reflection ---",
            f"Task        : {state.task}",
            f"Iterations  : {state.iteration}",
            f"Searches    : {len(searches)}",
            f"Observations: {len(observations)}",
            f"Completed   : {state.done}",
        ]

        if searches:
            lines.append("\nQueries used:")
            for s in searches:
                lines.append(f"  - {s.content}")

        return "\n".join(lines)
