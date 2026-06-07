import tempfile
from typing import Annotated

from langchain_core.tools import tool

from app.settings import settings
from app.tools.code_executor import BaseCodeExecutor

# Sandbox-side path where LLM-generated code must save output files.
SANDBOX_OUTPUT_PATH = "/tmp/sandbox_output"

# Magic bytes → file extension mapping
_MAGIC = [
    (b"%PDF", ".pdf"),
    (b"PK\x03\x04", ".xlsx"),   # zip-based: xlsx, docx, etc.
    (b"\xd0\xcf\x11\xe0", ".xls"),
]


def _detect_extension(data: bytes) -> str:
    for magic, ext in _MAGIC:
        if data.startswith(magic):
            return ext
    return ".bin"


def _wrap_for_output(code: str) -> str:
    """If the last statement is a bare expression, wrap it in print() so it appears in stdout."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        last_src = code.splitlines()[last.lineno - 1 : last.end_lineno]
        expr_code = "\n".join(last_src).strip()
        prefix = code[: code.rfind(expr_code)]
        return f"{prefix}print({expr_code})"
    return code


class DaytonaCodeExecutor(BaseCodeExecutor):
    """
    Executes code in a persistent Daytona sandbox.

    The sandbox is created on first use and reused for the lifetime of this object,
    so execute() and download() share the same filesystem state.
    Call close() when done to remove the sandbox.
    """

    def __init__(self):
        self._client = None
        self._sandbox = None

    def _ensure_sandbox(self):
        from daytona_sdk import Daytona, DaytonaConfig, CreateSandboxFromSnapshotParams, CodeLanguage
        if self._client is None:
            self._client = Daytona(DaytonaConfig(
                api_key=settings.daytona_api_key,
                server_url=settings.daytona_server_url,
            ))
        if self._sandbox is None:
            self._sandbox = self._client.create(
                CreateSandboxFromSnapshotParams(language=CodeLanguage.PYTHON)
            )

    def execute(self, code: str, language: str = "python") -> str:
        if not settings.daytona_api_key:
            return "Error: daytona_api_key is not set in settings."

        try:
            self._ensure_sandbox()
            wrapped = _wrap_for_output(code) if language.lower() == "python" else code
            response = self._sandbox.process.code_run(wrapped)
            result = response.result if hasattr(response, "result") else str(response)

            # If code saved a file to the well-known output path, download it
            try:
                local_path = self.download(SANDBOX_OUTPUT_PATH)
                return f"File saved to: {local_path}"
            except Exception:
                pass

            return result if result else "(no output)"
        except Exception as exc:
            return f"Execution error: {exc}"

    def download(self, sandbox_path: str) -> str:
        """
        Download a file from the sandbox at sandbox_path to a local temp file.
        Returns the local file path. Raises if the file does not exist.

        Example (from Python):
            from app.tools.daytona_executor import _executor
            local = _executor.download("/tmp/report.pdf")
            print(local)
        """
        self._ensure_sandbox()
        file_data: bytes | None = self._sandbox.fs.download_file(sandbox_path)
        if not file_data:
            raise FileNotFoundError(f"No file at '{sandbox_path}' in sandbox.")
        ext = _detect_extension(file_data)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_data)
            return tmp.name

    def close(self):
        """Remove the sandbox. Call this when the session is done."""
        if self._sandbox is not None and self._client is not None:
            try:
                self._client.remove(self._sandbox)
            except Exception:
                pass
            self._sandbox = None


# Module-level instance — swap to DockerCodeExecutor() or SubprocessCodeExecutor() here.
_executor = DaytonaCodeExecutor()


@tool
def sandbox_code_execution(
    code: Annotated[str, "Python code to execute in the sandbox. Use print() for any value you want to see — bare expressions produce no output."],
) -> str:
    """
    Execute Python code in an isolated Daytona sandbox and return stdout.
    IMPORTANT: This runs as a script, not a REPL. To see a value you MUST use print().
    Example: print(5 + 6)  →  returns '11'

    To generate a file (PDF, Excel, CSV, etc.):
    - Save the file to exactly '/tmp/sandbox_output' (no extension) inside the sandbox.
    - The file will be downloaded automatically and the local path returned.
    - Example: open('/tmp/sandbox_output', 'wb').write(pdf_bytes)
    """
    return _executor.execute(code)
