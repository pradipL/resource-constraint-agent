from typing import Dict

from app.tools.base import BaseTool, ToolResult


class ToolExecutor:
    def __init__(self, tools: Dict[str, BaseTool]):
        self.tools = tools

    def execute(self, tool_name: str, tool_input: str) -> ToolResult:
        tool = self.tools.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool '{tool_name}'. Available: {list(self.tools)}",
            )
        return tool.run(tool_input)
