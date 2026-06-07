import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.exceptions import BudgetExceeded, LLMLimitExceeded, TokenLimitReached
from app.agent.graph import ReActGraph
from app.agent.state import AgentState
from app.llm.cost_tracker import CostTracker
from app.tools.container_executor import sandbox_code_execution
from app.tools.rag_tool import rag_ingest, rag_search
from app.tools.web_search import web_search
from app.settings import settings

load_dotenv()

TRACES_DIR = Path(__file__).parent.parent / "traces"
CHECKPOINTS_DIR = Path(__file__).parent.parent / "checkpoints"


def _get_checkpointer(backend: str):
    if backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise ImportError(
                "SQLite backend requires: pip install langgraph-checkpoint-sqlite"
            ) from exc
        import sqlite3
        CHECKPOINTS_DIR.mkdir(exist_ok=True)
        conn = sqlite3.connect(
            str(CHECKPOINTS_DIR / "checkpoints.db"),
            check_same_thread=False,
        )
        return SqliteSaver(conn)

    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


def _initial_state(task: str) -> AgentState:
    return {
        "messages": [HumanMessage(content=task)],
        "iteration": 0,
        "plan": None,
        "completed_steps": [],
        "making_progress": None,
        "reflection_feedback": None,
    }


def _extract_final_answer(final_state: dict) -> str:
    messages = final_state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "No answer reached."


def _save_trace(task: str, thread_id: str, final_state: dict, tracker: CostTracker) -> str:
    TRACES_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = TRACES_DIR / f"trace_{ts}.json"

    messages = [
        {"type": m.__class__.__name__, "content": str(m.content)}
        for m in final_state.get("messages", [])
    ]
    path.write_text(json.dumps(
        {
            "task": task,
            "thread_id": thread_id,
            "messages": messages,
            "completed_steps": final_state.get("completed_steps", []),
            "cost": {
                "llm_calls": len(tracker.records),
                "prompt_tokens": tracker.total_prompt_tokens(),
                "completion_tokens": tracker.total_completion_tokens(),
                "total_tokens": tracker.total_tokens(),
                "estimated_cost_usd": tracker.total_cost_usd(),
            },
        },
        indent=2,
        default=str,
    ))
    print(f"Trace saved → {path}")
    return str(path)


def run_agent(
    task: str,
    thread_id: Optional[str] = None,
    backend: str = "memory",
    max_iterations: int = settings.max_llm_calls,
) -> dict:
    thread_id = thread_id or str(uuid.uuid4())

    # Add tools here — the LLM decides which ones to call
    tools = [web_search, rag_search, sandbox_code_execution]

    tracker = CostTracker()
    checkpointer = _get_checkpointer(backend)
    graph = ReActGraph(
        tools,
        checkpointer=checkpointer,
        max_llm_calls=max_iterations,
        cost_tracker=tracker,
    )
    config = {
        "configurable": {"thread_id": thread_id},
        # Each cycle: agent + tool_call + reflect (+ planner if replanning) = up to 4 nodes
        "recursion_limit": max_iterations * 4 + 5,
    }

    print(f"\n{'='*60}")
    print(f"Task      : {task}")
    print(f"Thread ID : {thread_id}  ← pass to --resume to continue")
    print(f"Backend   : {backend} checkpointer")
    print(f"Tools     : {[t.name for t in tools]}")
    print(f"{'='*60}")

    stopped_reason = None
    try:
        final_state = graph.invoke(_initial_state(task), config=config)
    except (LLMLimitExceeded, BudgetExceeded, TokenLimitReached) as exc:
        stopped_reason = str(exc)
        final_state = graph.get_state(config).values

    completed_steps = final_state.get("completed_steps", [])
    if stopped_reason:
        if completed_steps:
            steps_str = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(completed_steps))
            answer = f"Stopped early: {stopped_reason}\n\nSummary of work completed:\n\n{steps_str}"
        else:
            answer = f"Stopped early: {stopped_reason}\n\nNo steps were completed before stopping."
    else:
        answer = _extract_final_answer(final_state)

    print(f"\n{'='*60}")
    if stopped_reason:
        print(f"[STOPPED] {stopped_reason}")
    print("FINAL ANSWER:")
    print(answer)
    print(f"{'='*60}")
    print(f"LLM calls made  : {final_state.get('iteration', 0)}/{max_iterations}")
    print(f"Tool rounds     : {graph.current_depth}")
    print(f"\n{tracker.summary()}")

    trace_path = _save_trace(task, thread_id, final_state, tracker)

    return {
        "thread_id": thread_id,
        "answer": answer,
        "stopped_reason": stopped_reason,
        "completed_steps": completed_steps,
        "trace_path": trace_path,
        "cost": {
            "llm_calls": len(tracker.records),
            "prompt_tokens": tracker.total_prompt_tokens(),
            "completion_tokens": tracker.total_completion_tokens(),
            "total_tokens": tracker.total_tokens(),
            "estimated_cost_usd": tracker.total_cost_usd(),
        },
    }


def resume_agent(thread_id: str, backend: str = "memory") -> None:
    """Resume an interrupted run from its last LangGraph checkpoint."""
    tools = [web_search, sandbox_code_execution, rag_search]
    checkpointer = _get_checkpointer(backend)
    graph = ReActGraph(tools, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = graph.get_state(config)
    if not snapshot.values:
        print(f"No checkpoint found for thread '{thread_id}'.")
        return

    print(f"\nResuming thread '{thread_id}' …\n")
    final_state = graph.invoke(None, config=config)

    answer = _extract_final_answer(final_state)
    print(f"\n{'='*60}")
    print("FINAL ANSWER:")
    print(answer)
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom ReAct agent with LangGraph checkpointing")
    parser.add_argument("task", nargs="*", help="Task for the agent")
    parser.add_argument("--resume", metavar="THREAD_ID", help="Resume a prior run")
    parser.add_argument("--thread-id", help="Reuse a specific thread ID")
    parser.add_argument("--backend", choices=["memory", "sqlite"], default="memory")
    parser.add_argument("--max-iterations", type=int, default=settings.max_llm_calls)
    args = parser.parse_args()

    if args.resume:
        resume_agent(args.resume, backend=args.backend)
    else:
        task = " ".join(args.task) if args.task else input("Task: ").strip()
        if not task:
            task = "What are the latest breakthroughs in AI agents as of 2025?"
        result = run_agent(
            task,
            thread_id=args.thread_id,
            backend=args.backend,
            max_iterations=args.max_iterations,
        )
        print(f"\nThread ID: {result['thread_id']}")
