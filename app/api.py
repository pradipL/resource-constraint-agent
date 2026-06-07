import asyncio
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

_KB_PATH = Path(__file__).parent.parent / "app" / "tasks" / "knowledge_base2.txt"


def _already_ingested() -> bool:
    """Return True if the collection exists and already has points."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )
        collection_name = os.getenv("QDRANT_COLLECTION", "knowledge_base")
        names = {c.name for c in client.get_collections().collections}
        if collection_name not in names:
            return False
        return (client.get_collection(collection_name).points_count or 0) > 0
    except Exception:
        return False


def _run_ingest() -> None:
    if _already_ingested():
        print("[ingest] Collection already populated — skipping.", flush=True)
        return

    if not _KB_PATH.exists():
        print(f"[ingest] WARNING: {_KB_PATH} not found — skipping.", flush=True)
        return

    print(f"[ingest] Ingesting {_KB_PATH} ...", flush=True)
    try:
        from app.tools.rag_tool import ingest_text
        result = ingest_text(_KB_PATH.read_text(encoding="utf-8"), source="knowledge_base")
        print(f"[ingest] Done — {result}", flush=True)
    except Exception as exc:
        print(f"[ingest] ERROR: {exc}", file=sys.stderr, flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Block startup until ingest finishes — skips instantly on restarts
    await asyncio.get_running_loop().run_in_executor(None, _run_ingest)
    yield


app = FastAPI(title="Resource Agent API", version="1.0.0", lifespan=lifespan)


class RunRequest(BaseModel):
    task: str
    thread_id: str = 1
    backend: str = "memory"
    max_iterations: int = 10


class CostInfo(BaseModel):
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class RunResponse(BaseModel):
    thread_id: str
    answer: str
    stopped_reason: Optional[str]
    completed_steps: list[str]
    trace_path: str
    cost: CostInfo
    downloadable_path: Optional[str] = None


_FILE_SAVED_RE = re.compile(r"File saved to:\s*(\S+)")


def _extract_downloadable_path(result: dict) -> Optional[str]:
    """Return the first output file path found in completed_steps or the answer."""
    sources = result.get("completed_steps", []) + [result.get("answer", "")]
    for text in sources:
        m = _FILE_SAVED_RE.search(text or "")
        if m:
            return m.group(1)
    return None


@app.get("/download")
def download(path: str = Query(..., description="Absolute path of the file inside the container, e.g. /app/tmp/sandbox_output_123.pdf or /app/traces/trace_xyz.json")):
    file = Path(path).resolve()
    allowed = [Path("/app/tmp").resolve(), Path("/app/traces").resolve()]

    if not any(str(file).startswith(str(a)) for a in allowed):
        raise HTTPException(status_code=403, detail="Access denied: path must be inside /app/tmp or /app/traces")

    if not file.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    return FileResponse(path=str(file), filename=file.name, media_type="application/octet-stream")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest):
    from app.main import run_agent
    from app.settings import settings

    max_iter = req.max_iterations or settings.max_llm_calls

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_agent(
                task=req.task,
                thread_id=req.thread_id,
                backend=req.backend,
                max_iterations=max_iter,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return RunResponse(**result, downloadable_path=_extract_downloadable_path(result))
