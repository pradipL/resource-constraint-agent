from abc import ABC, abstractmethod


class BaseCodeExecutor(ABC):
    """
    Swappable backend for code execution.

    Implement this to support different runtimes (Daytona, Docker, subprocess, etc.).
    The LangChain @tool in each backend module delegates to an instance of this class,
    so switching backends requires no changes to the agent graph.
    """

    @abstractmethod
    def execute(self, code: str, language: str = "python") -> str:
        """
        Run `code` in the target runtime and return stdout/result as a string.
        On failure, return a descriptive error string (never raise).
        """
