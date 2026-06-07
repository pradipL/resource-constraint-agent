import operator
from typing import Annotated, List, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int  # number of LLM calls made so far; enforced in action_exists
    plan: Optional[str]  # numbered plan produced by the planner node before execution
    completed_steps: Annotated[List[str], operator.add]  # one compact record per tool call, appended by reflect
    making_progress: Optional[bool]  # set by reflect after each tool call; False triggers replanning
    reflection_feedback: Optional[str]  # reason extracted by reflect when no progress; passed to planner for replanning
    token_limit_reached: Optional[bool]  # set True when MAX_TOKEN_LIMIT is hit; triggers summary in agent
