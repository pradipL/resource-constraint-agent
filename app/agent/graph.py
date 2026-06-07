import os
from typing import Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph

from app.agent.exceptions import BudgetExceeded, LLMLimitExceeded, TokenLimitReached
from app.agent.prompts import (
    PROGRESS_TRACKER_SYSTEM_PROMPT,
    PROGRESS_TRACKER_USER_PROMPT,
    REFLECTION_ASSISTANT_PROMPT,
    REFLECTION_SYSTEM_PROMPT,
    REFLECTION_USER_PROMPT,
    REPLANNING_ASSISTANT_PROMPT,
    REPLANNING_SYSTEM_PROMPT,
    REPLANNING_USER_PROMPT,
    SYSTEM_PLANNING_PROMPT,
    SYSTEM_PROMPT,
    USER_PLANNING_PROMPT,
)
from app.agent.state import AgentState
from app.llm.cost_tracker import CostTracker
from app.settings import settings


_LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))


def _build_chat_model() -> tuple[BaseChatModel, str]:
    """Factory — returns (model, provider) based on LLM_PROVIDER env var."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs: dict = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "timeout": _LLM_TIMEOUT,
        }
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs), provider

    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        timeout=_LLM_TIMEOUT,
    ), provider


class ReActGraph:
    """
    Custom ReAct agent implemented as a self-contained class.

    Graph flow
    ----------
    START → planner → agent → [action_exists?]
                                   ├─ True  → tool_call → reflect → [should_replan?]
                                   │                                    ├─ True  → planner (replan) → agent → …
                                   │                                    └─ False → agent → …
                                   └─ False → END

    Nodes
    -----
    - planner   : LLM call (no tools) that produces a numbered plan.
                  On the first call it creates the initial plan from the task.
                  On subsequent calls (when making_progress=False) it receives
                  the completed steps + reflection feedback and creates a revised
                  plan using a different approach.
    - agent     : builds the full message context, detects the last allowed call
                  for synthesis mode, then delegates the actual LLM invocation to
                  call_llm.
    - tool_call : dispatches tool calls requested by the agent
    - reflect   : after each tool call, asks the LLM "Am I making progress?";
                  records the step in completed_steps and sets making_progress +
                  reflection_feedback in state

    Guards
    ------
    - action_exists : primary gate — returns False when LLM limit or budget is hit
    - agent         : secondary gate — raises LLMLimitExceeded / BudgetExceeded
                      as a safety net if action_exists is somehow bypassed
    - tool_call     : sets retry instruction only when a further LLM call is
                      still affordable; suppresses it otherwise
    - reflect       : defaults to making_progress=True when budget is near limit
                      so the last call goes to the agent rather than reflection
    """

    def __init__(
        self,
        tools: List[BaseTool],
        checkpointer=None,
        max_llm_calls: int = settings.max_llm_calls,
        max_cost_usd: float = 0.20,
        cost_tracker: CostTracker = None,
    ):
        self.tools: Dict[str, BaseTool] = {t.name: t for t in tools}
        self.cost_tracker = cost_tracker or CostTracker()
        self.max_llm_calls = max_llm_calls
        self.max_cost_usd = max_cost_usd
        self.max_completion_tokens = settings.max_completion_tokens
        self.model_max_output_tokens = settings.model_max_output_tokens
        self.max_token_limit = settings.max_token_limit  # 0 = disabled
        self.summary_token_buffer = settings.summary_token_buffer
        # Effective threshold: trigger token-limit handling early enough to
        # leave SUMMARY_TOKEN_BUFFER tokens for the progress-tracker LLM call.
        self._token_trigger = (
            max(0, self.max_token_limit - self.summary_token_buffer)
            if self.max_token_limit
            else 0
        )
        self.current_depth = 0

        self._ollama_model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
        self._ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        llm, self.provider = _build_chat_model()
        self.model = llm.bind_tools(tools)
        self.plain_llm = llm  # no tools — used for planning, reflection, replanning

        graph = StateGraph(AgentState)
        graph.add_node("planner", self.plan_task)
        graph.add_node("agent", self.agent)
        graph.add_node("tool_call", self.tool_call)
        graph.add_node("reflect", self.reflect)
        graph.add_conditional_edges(
            "agent", self.action_exists, {True: "tool_call", False: END}
        )
        graph.add_conditional_edges(
            "reflect", self.should_replan, {True: "planner", False: "agent"}
        )
        graph.add_edge("planner", "agent")
        graph.add_edge("tool_call", "reflect")
        graph.set_entry_point("planner")
        self.graph = graph.compile(checkpointer=checkpointer)
        print("\n[graph] Compiled graph structure:")
        print(self.graph.get_graph().draw_mermaid())

    # ── LLM helper ─────────────────────────────────────────────────────────

    def call_llm(self, messages: list, use_tools: bool = True) -> AIMessage:
        """
        Minimal LLM invocation helper.
        Selects the tool-bound or plain model, invokes it, records cost, returns
        the response. All context-building lives in the calling node.
        """
        from openai import BadRequestError

        total_used = self.cost_tracker.total_tokens()

        # Hard stop when used tokens hit the buffered trigger (leaves room for summary).
        if self._token_trigger and total_used >= self._token_trigger:
            raise TokenLimitReached(total_used, self._token_trigger)

        # Estimate prompt tokens for this call (chars / 4 is a standard approximation).
        prompt_estimate = sum(
            len(m.content) // 4
            for m in messages
            if hasattr(m, "content") and isinstance(m.content, str)
        )

        # Headroom is computed against the buffered trigger so normal calls never
        # consume the SUMMARY_TOKEN_BUFFER that is reserved for the summary response.
        # If the prompt estimate alone would exhaust the budget, bail now — a
        # max_tokens=1 call still records full prompt tokens and overshoots the limit.
        if self._token_trigger:
            raw_headroom = self._token_trigger - total_used - prompt_estimate
            if raw_headroom <= 0:
                raise TokenLimitReached(total_used, self._token_trigger)
            token_headroom = raw_headroom
        else:
            token_headroom = self.model_max_output_tokens

        session_remaining = max(1, self.max_completion_tokens - self.cost_tracker.total_completion_tokens())
        remaining_tokens = min(session_remaining, self.model_max_output_tokens, token_headroom)
        print(
            f"[call_llm] used={total_used} | prompt_est={prompt_estimate} "
            f"| headroom={token_headroom} | max_tokens={remaining_tokens}"
        )

        if self.provider == "ollama":
            # ChatOllama ignores num_predict via .bind() — must be set at construction.
            from langchain_ollama import ChatOllama
            new_llm = ChatOllama(
                model=self._ollama_model_name,
                base_url=self._ollama_base_url,
                num_predict=remaining_tokens,
                timeout=_LLM_TIMEOUT,
            )
            print(f"[call_llm] Ollama num_predict={remaining_tokens}")
            model = new_llm.bind_tools(list(self.tools.values())) if use_tools else new_llm
        else:
            model = self.model if use_tools else self.plain_llm
            model = model.bind(max_tokens=remaining_tokens)
            print(f"[call_llm] OpenAI max_tokens={remaining_tokens}")

        try:
            response = model.invoke(messages)
        except BadRequestError as exc:
            msg = str(exc)
            if "max_tokens" in msg or "output limit" in msg:
                print("[call_llm] Token limit reached — exiting gracefully.")
                raise TokenLimitReached() from exc
            raise
        except Exception as exc:
            name = type(exc).__name__
            msg = str(exc).lower()
            if any(k in msg for k in ("timeout", "timed out", "connection", "connect error")):
                raise RuntimeError(f"LLM call timed out or unreachable ({name}): {exc}") from exc
            raise
        meta = getattr(response, "response_metadata", {}) or {}
        usage = getattr(response, "usage_metadata", {}) or {}
        # usage_metadata is provider-agnostic; fall back to Ollama-specific keys
        self.cost_tracker.record(
            model=meta.get("model") or meta.get("model_name") or os.getenv("OLLAMA_MODEL", "unknown"),
            prompt_tokens=usage.get("input_tokens") or meta.get("prompt_eval_count", 0),
            completion_tokens=usage.get("output_tokens") or meta.get("eval_count", 0),
        )
        # Post-call check: actual tokens are now recorded; raise if we crossed the
        # trigger despite passing the pre-call estimate (estimate uses chars/4 which
        # can undercount, letting a call through that overshoots the real limit).
        new_total = self.cost_tracker.total_tokens()
        if self._token_trigger and new_total >= self._token_trigger:
            print(
                f"[call_llm] Post-call overshoot: {new_total:,} >= trigger {self._token_trigger:,} "
                f"— raising TokenLimitReached"
            )
            raise TokenLimitReached(new_total, self._token_trigger)
        return response

    def _token_limit_summary(self, state: AgentState) -> dict:
        """
        Called when MAX_TOKEN_LIMIT is hit. Makes one final LLM call (bypassing the
        token limit check) using the progress tracker prompt to summarise what was
        completed and what remains. Returns with no tool calls so action_exists
        routes to END naturally.
        """
        completed = state.get("completed_steps", [])
        plan = state.get("plan", "")
        used = self.cost_tracker.total_tokens()
        print(
            f"[token_limit] Token trigger reached ({used:,}/{self._token_trigger:,}, "
            f"buffer={self.summary_token_buffer}) — generating progress summary "
            f"({len(completed)} step(s) completed)."
        )

        steps_str = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(completed)) if completed else "  None"
        plan_section = f"<plan>\n{plan}\n</plan>\n\n" if plan else ""

        summary_messages = [
            SystemMessage(content=PROGRESS_TRACKER_SYSTEM_PROMPT),
            AIMessage(content=f"Here are the steps completed during execution:\n{steps_str}"),
            HumanMessage(content=PROGRESS_TRACKER_USER_PROMPT.format(
                plan_section=plan_section,
                progress_made=steps_str,
            )),
        ]

        # Bypass token limit — invoke plain_llm directly without call_llm checks.
        try:
            response = self.plain_llm.invoke(summary_messages)
            content = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            content = f"Token limit reached. Could not generate summary: {exc}\n\nCompleted steps:\n{steps_str}"

        return {
            "messages": [AIMessage(content=content)],
            "token_limit_reached": True,
        }

    # ── Nodes ──────────────────────────────────────────────────────────────

    def plan_task(self, state: AgentState) -> dict:
        """
        Produces a numbered plan from the user task.

        Called twice in two different contexts:
        1. Initial planning (no prior plan / no completed steps) — uses PLANNING_PROMPT.
        2. Replanning (making_progress=False) — uses REPLANNING_PROMPT with the
           completed steps and reflection feedback so the new plan tries a
           different approach.
        """
        iteration = state.get("iteration", 0)
        current_cost = self.cost_tracker.total_cost_usd()

        if iteration >= self.max_llm_calls:
            raise LLMLimitExceeded(self.max_llm_calls)
        if current_cost >= self.max_cost_usd:
            raise BudgetExceeded(current_cost, self.max_cost_usd)
        current_tokens = self.cost_tracker.total_tokens()
        if self._token_trigger and current_tokens >= self._token_trigger:
            raise TokenLimitReached(current_tokens, self._token_trigger)

        messages = state["messages"]
        existing_plan = state.get("plan")
        completed_steps = state.get("completed_steps", [])
        feedback = state.get("reflection_feedback", "")

        is_replanning = bool(existing_plan and completed_steps)

        if is_replanning:
            task = next(
                (msg.content for msg in messages if isinstance(msg, HumanMessage)),
                "",
            )
            completed_str = "\n".join(
                f"  {i + 1}. {step}" for i, step in enumerate(completed_steps)
            )
            planning_messages = [
                SystemMessage(content=REPLANNING_SYSTEM_PROMPT),
                AIMessage(content=REPLANNING_ASSISTANT_PROMPT.format(
                    task=task,
                    plan=existing_plan,
                    completed_steps=completed_str,
                )),
                HumanMessage(content=REPLANNING_USER_PROMPT.format(
                    feedback=feedback or "No specific reason given.",
                )),
            ]
            label = "replanner"
        else:
            task = next(
                (msg.content for msg in messages if isinstance(msg, HumanMessage)),
                "",
            )
            planning_messages = [
                SystemMessage(content=SYSTEM_PLANNING_PROMPT),
                HumanMessage(content=USER_PLANNING_PROMPT.format(user_request=task)),
            ]
            label = "planner"

        try:
            response = self.call_llm(planning_messages, use_tools=True)
        except TokenLimitReached:
            print("[plan_task] Token limit reached during planning — routing to summary.")
            return {"token_limit_reached": True}

        plan_text = response.content if hasattr(response, "content") else str(response)
        new_iteration = iteration + 1
        tag = "REVISED PLAN" if is_replanning else "PLAN"
        print(f"\n[{label}] LLM call {new_iteration}/{self.max_llm_calls}:")
        print(plan_text)

        return {
            "messages": [AIMessage(content=f"{tag}:\n{plan_text}")],
            "iteration": new_iteration,
            "plan": plan_text,
            "making_progress": True,  # reset after every (re)plan
            "reflection_feedback": None,
        }

    def agent(self, state: AgentState) -> dict:
        """
        Agent node: enforces budget, builds the full message context, detects
        whether this is the last allowed LLM call and switches to synthesis mode
        if so, then delegates to call_llm.

        Synthesis mode — triggered on the last call when work has been done:
          Switches to plain_llm (no tool binding) and injects a FINAL RESPONSE
          instruction with the completed steps so the LLM summarises progress
          instead of requesting more tools.

        Normal mode:
          Uses the tool-bound model. Injects a HumanMessage after a plan AIMessage
          so the LLM opens a proper execution turn (avoids AI→AI echo).
        """
        iteration = state.get("iteration", 0)
        current_cost = self.cost_tracker.total_cost_usd()

        # Safety-net guard — primary guard is action_exists
        if iteration >= self.max_llm_calls:
            raise LLMLimitExceeded(self.max_llm_calls)
        if current_cost >= self.max_cost_usd:
            raise BudgetExceeded(current_cost, self.max_cost_usd)
        current_tokens = self.cost_tracker.total_tokens()
        if state.get("token_limit_reached") or (self._token_trigger and current_tokens >= self._token_trigger):
            return self._token_limit_summary(state)

        messages = list(state["messages"])
        plan = state.get("plan", "")
        completed_steps = state.get("completed_steps", [])

        system_prompt = SYSTEM_PROMPT

        is_last_call = iteration >= self.max_llm_calls - 1
        cost_near_limit = current_cost >= self.max_cost_usd * 0.9

        if (is_last_call or cost_near_limit) and completed_steps:
            # ── Synthesis mode ──────────────────────────────────────────────
            # Last allowed call: compare plan vs. progress and summarise.
            steps_str = "\n".join(
                f"  {i + 1}. {s}" for i, s in enumerate(completed_steps)
            )
            synthesis_messages = [
                SystemMessage(content=PROGRESS_TRACKER_SYSTEM_PROMPT),
                AIMessage(content=f"Here are the steps completed during execution:\n{steps_str}"),
                HumanMessage(content=PROGRESS_TRACKER_USER_PROMPT.format(
                    plan_section=plan,
                    progress_made=steps_str,
                )),
            ]
            try:
                response = self.call_llm(synthesis_messages, use_tools=False)
            except TokenLimitReached:
                return self._token_limit_summary(state)
        else:
            # ── Normal execution mode ───────────────────────────────────────
            system_content = system_prompt
            if plan:
                system_content = f"PLAN TO FOLLOW:\n{plan}\n\n{system_prompt}"

            # After (re)planning the last message is an AIMessage with the plan
            # text and no tool calls. The LLM would see AI→AI and echo the plan
            # as plain text instead of executing. A HumanMessage opens a proper
            # conversational turn so the LLM responds with tool calls.
            last = messages[-1] if messages else None
            if (
                isinstance(last, AIMessage)
                and not last.tool_calls
                and isinstance(last.content, str)
                and (
                    last.content.startswith("PLAN:")
                    or last.content.startswith("REVISED PLAN:")
                )
            ):
                messages = messages + [
                    HumanMessage(
                        content="Execute step 1 of the plan now by calling the appropriate tool and plan it in a way that fixes previous issues."
                    )
                ]

            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=system_content)] + messages
            try:
                response = self.call_llm(messages, use_tools=True)
            except TokenLimitReached:
                return self._token_limit_summary(state)

        new_iteration = iteration + 1
        print(
            f"\n[agent] LLM call {new_iteration}/{self.max_llm_calls} "
            f"| cost so far: ${self.cost_tracker.total_cost_usd():.6f}/${self.max_cost_usd:.2f}"
        )
        return {"messages": [response], "iteration": new_iteration}

    def tool_call(self, state: AgentState) -> dict:
        """
        Dispatch every tool call the LLM requested.

        Retry logic
        -----------
        A retry instruction is added to the ToolMessage only when the agent
        can actually act on it — i.e., there is still LLM call budget AND the
        cost ceiling has not been reached.

        If the error itself is LLMLimitExceeded (e.g., a tool that wraps an
        LLM internally hit its limit), retry is suppressed regardless.
        """
        tool_calls = state["messages"][-1].tool_calls
        results = []

        # After this tool call the LLM will need one more call to act on a retry.
        # can_retry is True only when that next call is still within both limits.
        current_iteration = state.get("iteration", 0)
        can_retry = (
            current_iteration < self.max_llm_calls - 1
            and self.cost_tracker.total_cost_usd() < self.max_cost_usd
        )

        token_limit_hit = False

        for idx, t in enumerate(tool_calls):
            print(f"\n[tool_call] → {t['name']}({t['args']})")

            if t["name"] not in self.tools:
                content = (
                    f"Error: tool '{t['name']}' does not exist. "
                    f"Available tools: {list(self.tools)}. "
                    f"Retry using one of the available tools."
                )
            else:
                tool = self.tools[t["name"]]
                try:
                    current_tokens = self.cost_tracker.total_tokens()
                    if self._token_trigger and current_tokens >= self._token_trigger:
                        raise TokenLimitReached(current_tokens, self._token_trigger)

                    result = tool.invoke(t["args"])

                    if isinstance(result, dict) and result.get("status") == "timeout":
                        print(f"[tool_call] ⚠ '{t['name']}' {result['message']}")
                        suffix = (
                            " Try a simpler or more specific approach."
                            if can_retry
                            else " Cannot retry: budget limit reached."
                        )
                        content = f"{result['message']}.{suffix}"
                    else:
                        content = str(result)
                        if (
                            t["name"] == "sandbox_code_execution"
                            and content.startswith("Error")
                            and can_retry
                        ):
                            content += "\nFix the error in your code and retry with corrected code."

                except TokenLimitReached as exc:
                    print(f"[tool_call] Token limit reached executing '{t['name']}': {exc}")
                    results.append(ToolMessage(
                        tool_call_id=t["id"],
                        name=t["name"],
                        content=f"Token limit reached while executing '{t['name']}'. Stopping execution.",
                    ))
                    for remaining_t in tool_calls[idx + 1:]:
                        results.append(ToolMessage(
                            tool_call_id=remaining_t["id"],
                            name=remaining_t["name"],
                            content="Skipped: token limit reached.",
                        ))
                    token_limit_hit = True
                    break

                except LLMLimitExceeded as exc:
                    # Tool itself hit an LLM limit — never ask to retry
                    content = f"Cannot retry: {exc}"

                except Exception as exc:
                    if can_retry:
                        if t["name"] == "sandbox_code_execution":
                            content = (
                                f"Code execution error: {exc}\n"
                                f"Fix the error in your code and retry with corrected code."
                            )
                        else:
                            content = (
                                f"Error calling '{t['name']}': {exc}\n"
                                f"Retry with a different or more specific query."
                            )
                    else:
                        reason = (
                            f"LLM call limit ({self.max_llm_calls}) reached"
                            if current_iteration >= self.max_llm_calls - 1
                            else f"cost budget (${self.max_cost_usd:.2f}) reached"
                        )
                        content = (
                            f"Error calling '{t['name']}': {exc}\n"
                            f"Cannot retry: {reason}."
                        )

            if not token_limit_hit:
                print(f"[tool_call] result (truncated): {content[:300]}")
                results.append(
                    ToolMessage(
                        tool_call_id=t["id"],
                        name=t["name"],
                        content=content,
                    )
                )

        self.current_depth += 1
        print(f"[tool_call] depth now: {self.current_depth}")

        if token_limit_hit:
            return {"messages": results, "token_limit_reached": True}

        return {"messages": results}

    def reflect(self, state: AgentState) -> dict:
        """
        Evaluate progress after each tool call.

        Asks the LLM: "Am I making progress toward solving the task?"
        Records the last action(s) in completed_steps regardless of the answer.
        Sets making_progress=False + reflection_feedback when no progress is
        detected, which routes the graph back to planner for a revised plan.

        The LLM call is skipped when only one budget slot remains; that slot is
        reserved for the agent's final response.
        """
        iteration = state.get("iteration", 0)
        current_cost = self.cost_tracker.total_cost_usd()

        step_records = self._extract_last_step_records(state)

        # Reserve the last LLM slot for the agent — skip reflection if we're close or token limit hit
        if state.get("token_limit_reached") or iteration >= self.max_llm_calls - 1 or current_cost >= self.max_cost_usd:
            reason = "token limit reached" if state.get("token_limit_reached") else "budget near limit"
            print(f"[reflect] Skipping LLM evaluation ({reason}), assuming progress.")
            return {
                "messages": [],
                "making_progress": True,
                "completed_steps": step_records,
            }

        completed_so_far = state.get("completed_steps", [])
        completed_str = (
            "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(completed_so_far))
            or "  None yet"
        )
        last_actions_str = (
            "\n".join(f"  - {r}" for r in step_records)
            or "  None"
        )

        try:
            response = self.call_llm(
                [
                    SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
                    AIMessage(content=REFLECTION_ASSISTANT_PROMPT.format(
                        plan=state.get("plan", "No plan available"),
                        completed_steps=completed_str,
                    )),
                    HumanMessage(content=REFLECTION_USER_PROMPT.format(
                        last_actions=last_actions_str,
                    )),
                ],
                use_tools=False,
            )
        except TokenLimitReached:
            print("[reflect] Token limit reached during reflection — skipping, routing to summary.")
            return {
                "messages": [],
                "making_progress": True,
                "completed_steps": step_records,
                "token_limit_reached": True,
            }

        new_iteration = iteration + 1

        reflection_text = (
            response.content if hasattr(response, "content") else str(response)
        )
        upper = reflection_text.upper()
        making_progress = "PROGRESS: NO" not in upper  # default True if format unclear

        # Extract the REASON line for the planner to use when replanning
        feedback = ""
        for line in reflection_text.splitlines():
            if line.strip().upper().startswith("REASON:"):
                feedback = line.split(":", 1)[-1].strip()
                break

        print(
            f"\n[reflect] LLM call {new_iteration}/{self.max_llm_calls} "
            f"| Making progress: {making_progress}"
        )
        print(f"[reflect] {reflection_text[:300]}")

        next_step_msg = HumanMessage(
            content=(
                f"[Reflection]: {reflection_text}\n\n"
                "The previous step is complete. Proceed to the next step of the plan "
                "by calling the appropriate tool."
            )
        )
        return {
            "messages": [next_step_msg],
            "making_progress": making_progress,
            "reflection_feedback": feedback if not making_progress else None,
            "completed_steps": step_records,
            "iteration": new_iteration,
        }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _last_round_tool_messages(self, state: AgentState) -> list:
        """Return the ToolMessages produced in the most recent tool-call round."""
        messages = state.get("messages", [])
        tool_msgs: list = []
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_msgs.insert(0, msg)
            elif isinstance(msg, AIMessage) and msg.tool_calls:
                break
            elif tool_msgs:
                break
        return tool_msgs

    def _extract_last_step_records(self, state: AgentState) -> list:
        """
        Walk the message list backwards to find the last AIMessage with tool_calls
        and its corresponding ToolMessages; return one compact string per call.
        """
        messages = state.get("messages", [])
        tool_results: list = []
        last_ai_msg = None

        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                tool_results.insert(0, msg)
            elif isinstance(msg, AIMessage) and msg.tool_calls:
                last_ai_msg = msg
                break
            elif tool_results:
                # Non-tool message encountered before finding the AI caller — stop
                break

        if not last_ai_msg:
            return []

        records = []
        for tc in last_ai_msg.tool_calls:
            result_msg = next(
                (r for r in tool_results if r.tool_call_id == tc["id"]), None
            )
            result_preview = result_msg.content[:800] if result_msg else "no result"
            records.append(f"{tc['name']}({tc['args']}) → {result_preview}")

        return records

    # ── Routing ────────────────────────────────────────────────────────────

    def action_exists(self, state: AgentState) -> bool:
        """
        Primary guard. Returns False (→ END) when:
          - LLM call count has reached max_llm_calls, OR
          - accumulated cost has reached max_cost_usd, OR
          - the last AI message has no pending tool calls.
        """
        iteration = state.get("iteration", 0)
        current_cost = self.cost_tracker.total_cost_usd()

        if iteration >= self.max_llm_calls:
            print(f"\n[agent] Stopping: LLM call limit ({self.max_llm_calls}) reached.")
            return False

        if current_cost >= self.max_cost_usd:
            print(
                f"\n[agent] Stopping: cost budget "
                f"(${self.max_cost_usd:.2f}) exceeded at ${current_cost:.6f}."
            )
            return False

        if state.get("token_limit_reached"):
            print("\n[agent] Token limit reached — producing progress summary.")
            return False

        current_tokens = self.cost_tracker.total_tokens()
        if self._token_trigger and current_tokens >= self._token_trigger:
            print(
                f"\n[agent] Stopping: token trigger ({self._token_trigger:,}, "
                f"limit={self.max_token_limit:,} buffer={self.summary_token_buffer}) "
                f"reached at {current_tokens:,} tokens."
            )
            return False

        messages = state.get("messages", [])
        if not messages:
            return False
        last = messages[-1]
        return isinstance(last, AIMessage) and bool(last.tool_calls)

    def should_replan(self, state: AgentState) -> bool:
        """Route back to planner when reflect detected no progress."""
        return state.get("making_progress") is False

    # ── Public API ─────────────────────────────────────────────────────────

    def invoke(self, state: AgentState, config: dict):
        return self.graph.invoke(state, config)

    def get_state(self, config: dict):
        return self.graph.get_state(config)
