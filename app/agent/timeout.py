import functools
from concurrent.futures import ThreadPoolExecutor, TimeoutError


def tool_timeout(seconds):
    """
    Decorator that enforces a wall-clock time limit on a tool function.

    On timeout the wrapper returns a sentinel dict instead of raising so the
    caller (graph.py tool_call) can format a clean ToolMessage and let the
    agent decide whether to retry or replan.

    Usage (always place UNDER @tool so LangChain inspects the original signature):

        @tool
        @tool_timeout(30)
        def web_search(query: str) -> str:
            ...
    """
    def decorator(func):
        @functools.wraps(func)  # preserve __name__, __doc__, __annotations__ for @tool
        def wrapper(*args, **kwargs):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except TimeoutError:
                    return {
                        "status": "timeout",
                        "message": f"{func.__name__} timed out after {seconds}s",
                    }
        return wrapper
    return decorator
