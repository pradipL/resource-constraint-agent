import io
import os
import tarfile
import time
from pathlib import Path
from typing import Annotated

# Local folder where generated files are saved.
_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "tmp"
_OUTPUT_DIR.mkdir(exist_ok=True)

from langchain_core.tools import tool

from app.agent.timeout import tool_timeout
from app.tools.code_executor import BaseCodeExecutor

# Docker image used for all sandboxed execution.
# Pre-built with: reportlab, fpdf2, openpyxl, xlsxwriter, pandas, matplotlib, pillow, python-docx
_IMAGE = "resource-agent-sandbox:latest"

# Path inside the container where generated output files must be saved.
CONTAINER_OUTPUT_PATH = "/tmp/sandbox_output"

# Magic bytes → file extension
_MAGIC = [
    (b"%PDF", ".pdf"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"\xd0\xcf\x11\xe0", ".xls"),
]

# ZIP-based formats (PK magic) — disambiguate by internal path
_ZIP_MAGIC = b"PK\x03\x04"
_ZIP_EXT_HINTS = [
    ("word/", ".docx"),
    ("ppt/", ".pptx"),
    ("xl/", ".xlsx"),
]


def _detect_extension(data: bytes) -> str:
    if data.startswith(_ZIP_MAGIC):
        # Scan the raw bytes for Office Open XML markers
        for hint, ext in _ZIP_EXT_HINTS:
            if hint.encode() in data[:4096]:
                return ext
        return ".xlsx"  # generic ZIP-based spreadsheet fallback

    for magic, ext in _MAGIC:
        if data.startswith(magic):
            return ext

    # CSV and other text formats carry no magic bytes — detect by decodability
    try:
        text = data.decode("utf-8")
        first_line = text.splitlines()[0] if text.strip() else ""
        if "," in first_line or "\t" in first_line:
            return ".csv"
        return ".txt"
    except UnicodeDecodeError:
        return ".bin"


def _wrap_for_output(code: str) -> str:
    """Wrap a bare last expression in print() so it appears on stdout.
    Skips wrapping if the last expression is already a print() call."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[-1], ast.Expr):
        return code
    last = tree.body[-1]
    # Already a print() call — don't double-wrap
    if (
        isinstance(last.value, ast.Call)
        and isinstance(last.value.func, ast.Name)
        and last.value.func.id == "print"
    ):
        return code
    # Use ast.unparse to reconstruct the expression without inline comments,
    # then replace only the last line(s) with the wrapped version.
    expr_code = ast.unparse(last.value)
    prefix_lines = code.splitlines()[: last.lineno - 1]
    prefix = "\n".join(prefix_lines)
    if prefix:
        prefix += "\n"
    return f"{prefix}print({expr_code})"


class ContainerCodeExecutor(BaseCodeExecutor):
    """
    Executes code inside a Docker container sandbox.

    Each execute() call spins up a fresh container, runs the code, captures
    stdout, optionally downloads a generated file, then removes the container.
    """

    _LANG_IMAGE = {
        "python": "resource-agent-sandbox:latest",
        "javascript": "node:20-slim",
        "typescript": "node:20-slim",
    }
    _LANG_CMD = {
        "python": ["python", "-c"],
        "javascript": ["node", "-e"],
        "typescript": ["npx", "ts-node", "-e"],
    }

    def _client(self):
        import docker
        return docker.from_env()

    def execute(self, code: str, language: str = "python") -> str:
        lang = language.lower()
        image = self._LANG_IMAGE.get(lang, _IMAGE)
        cmd_prefix = self._LANG_CMD.get(lang, ["python", "-c"])

        if lang == "python":
            code = _wrap_for_output(code)

        try:
            client = self._client()
            container = client.containers.run(
                image=image,
                command=cmd_prefix + [code],
                remove=False,
                detach=True,
                mem_limit="256m",
                network_disabled=True,
                read_only=False,
            )
            result = container.wait(timeout=30)
            stdout = container.logs(stdout=True, stderr=False).decode().strip()
            stderr = container.logs(stdout=False, stderr=True).decode().strip()

            # Surface errors before attempting file download
            if result["StatusCode"] != 0:
                container.remove(force=True)
                return f"Error (exit {result['StatusCode']}): {stderr or stdout or '(no output)'}"

            # Check if code saved a file to the well-known output path
            local_path = self.download(container, CONTAINER_OUTPUT_PATH)
            print('local path is', local_path)

            container.remove(force=True)

            if local_path:
                return f"File saved to: {local_path}"

            # Exit 0 but no file — surface stderr so the agent knows what went wrong
            if stderr:
                return f"Code exited successfully but no file was saved. Stderr:\n{stderr}"

            return stdout if stdout else "(no output)"

        except Exception as exc:
            return f"Execution error: {exc}"

    def download(self, container_or_id, sandbox_path: str) -> str | None:
        """
        Download a file from a container at sandbox_path to a local temp file.
        Returns the local path if the file exists, or None if it does not.

        Can be called with a running container object or a container ID string.

        Example:
            from app.tools.container_executor import _executor
            local = _executor.download(container, "/tmp/report.pdf")
        """
        try:
            import docker
            client = self._client()
            container = (
                container_or_id
                if not isinstance(container_or_id, str)
                else client.containers.get(container_or_id)
            )
            bits, _ = container.get_archive(sandbox_path)
            buf = io.BytesIO(b"".join(bits))
            with tarfile.open(fileobj=buf) as tar:
                member = tar.getmembers()[0]
                f = tar.extractfile(member)
                if f is None:
                    return None
                file_data = f.read()

            if not file_data:
                return None

            ext = _detect_extension(file_data)
            local_path = _OUTPUT_DIR / f"sandbox_output_{int(time.time())}{ext}"
            local_path.write_bytes(file_data)
            print("Downloaded file from sandbox to:", local_path)
            return str(local_path)

        except Exception as e:
            print(f'Error occurred while downloading file: {e}')
            return None


_executor = ContainerCodeExecutor()


@tool
@tool_timeout(60)
def sandbox_code_execution(
    code: Annotated[str, "Python code to run in the sandbox. Use print() for any value you want to see — bare expressions produce no output."],
) -> str:
    """
    Execute Python code in an isolated Docker container and return stdout.
    IMPORTANT: This runs as a script, not a REPL. To see a value you MUST use print().
    Wrap all code in try/except and print errors so the agent gets feedback on failures.
    Example: print(5 + 6)  →  returns '11'

    To generate a file (PDF, Excel, CSV, etc.):
    - Write the file to EXACTLY '/tmp/sandbox_output' (no extension) inside the container.
    - The file will be downloaded automatically and the local path returned.
    - Available libraries: reportlab, fpdf2, openpyxl, xlsxwriter, pandas, matplotlib, pillow, python-docx

    Correct patterns per file type:
      PDF (fpdf2):
        pdf = FPDF()
        ...
        with open('/tmp/sandbox_output', 'wb') as f:
            f.write(pdf.output())

      PDF (reportlab):
        from reportlab.pdfgen import canvas
        c = canvas.Canvas('/tmp/sandbox_output')
        ...
        c.save()

      Excel (openpyxl):
        wb.save('/tmp/sandbox_output')

      CSV / text:
        with open('/tmp/sandbox_output', 'w') as f:
            f.write(content)

    NEVER use pdf.output('/tmp/sandbox_output') — use pdf.output() and write the bytes manually.
    """
    return _executor.execute(code)
