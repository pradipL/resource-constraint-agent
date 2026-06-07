import os

from langchain_core.tools import tool
from tavily import TavilyClient

from app.agent.timeout import tool_timeout


@tool
@tool_timeout(30)
def web_search(query: str) -> str:
    """Search the web for current information. Input: a plain search query string."""
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(query.strip(), max_results=5)
    except Exception as exc:
        return f"Web search failed: {type(exc).__name__}: {exc}"
    results = response.get("results", [])

    if not results:
        return "No results found for that query."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.get('title', 'No title')}")
        lines.append(f"    URL: {r.get('url', '')}")
        lines.append(f"    {r.get('content', '')}")
        lines.append("")

    return "\n".join(lines).strip()
