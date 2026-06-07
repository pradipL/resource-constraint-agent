import os
import uuid
from typing import Annotated

from langchain_core.tools import tool

from app.agent.timeout import tool_timeout

_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_base")
_MAX_CHUNK_SIZE = 1000  # only used as a safety limit if a section is huge


_QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "30"))


def _qdrant_client():
    from qdrant_client import QdrantClient

    return QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        timeout=_QDRANT_TIMEOUT,
    )


def _embedder():
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        kwargs = {
            "model": os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            "api_key": os.getenv("OPENAI_API_KEY", ""),
        }
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAIEmbeddings(**kwargs)

    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )


def _chunk_text(text: str) -> list[str]:
    """
    Separator-first chunking:
      1. Split on \\n\\n — each paragraph/section is its own chunk.
      2. If a section exceeds _MAX_CHUNK_SIZE, split further on \\n.
    No merging of small chunks — size never drives splitting.
    """
    chunks = []
    for section in text.split("\n\n"):
        section = section.strip()
        if not section:
            continue
        if len(section) <= _MAX_CHUNK_SIZE:
            chunks.append(section)
        else:
            # Section too large — split by single newline
            current_lines = []
            current_len = 0
            for line in section.split("\n"):
                if current_len + len(line) > _MAX_CHUNK_SIZE and current_lines:
                    chunks.append("\n".join(current_lines))
                    current_lines = []
                    current_len = 0
                current_lines.append(line)
                current_len += len(line) + 1
            if current_lines:
                chunks.append("\n".join(current_lines))
    return chunks or [text]


def _ensure_collection(client, dim: int) -> None:
    from qdrant_client.models import Distance, VectorParams

    try:
        existing = {c.name for c in client.get_collections().collections}
        if _COLLECTION not in existing:
            client.create_collection(
                collection_name=_COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
    except Exception as exc:
        raise RuntimeError(f"Qdrant unreachable or timed out: {exc}") from exc


def ingest_text(content: str, source: str = "ingested") -> str:
    """Chunk, embed, and upsert text into Qdrant. Callable directly or via the agent tool."""
    from qdrant_client.models import PointStruct

    if not content or not content.strip():
        return "Error: content is empty."

    chunks = _chunk_text(content.strip())

    try:
        embedder = _embedder()
        vectors = embedder.embed_documents(chunks)
    except Exception as exc:
        return f"Embedding failed: {exc}"

    try:
        client = _qdrant_client()
        _ensure_collection(client, len(vectors[0]))
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"text": chunk, "source": source, "chunk_index": idx},
            )
            for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        client.upsert(collection_name=_COLLECTION, points=points)
    except RuntimeError:
        raise
    except Exception as exc:
        return f"Qdrant write failed: {exc}"
    return (
        f"Stored {len(chunks)} chunk(s) from '{source}' into the knowledge base "
        f"(collection: {_COLLECTION})."
    )


@tool
@tool_timeout(120)
def rag_ingest(
    content: Annotated[str, "Text content to store in the knowledge base"],
    source: Annotated[
        str, "Label for this content (e.g. 'web_result', 'document_name')"
    ] = "ingested",
) -> str:
    """
    Chunk and embed text into the Qdrant vector database for later semantic retrieval.
    Call this to persist information so it can be found with rag_search.
    Returns a confirmation with the number of chunks stored.
    """
    return ingest_text(content, source)


@tool
@tool_timeout(60)
def rag_search(
    query: Annotated[str, "The search question or topic"],
    top_k: Annotated[int, "Number of results to return (1–10, default 4)"] = 4,
) -> str:
    """
    Search the Qdrant knowledge base for text semantically similar to a query.
    Returns the most relevant chunks with their similarity scores and sources.
    """
    if not query or not query.strip():
        return "Error: query is empty."

    top_k = max(1, min(3, top_k))

    try:
        client = _qdrant_client()
        existing = {c.name for c in client.get_collections().collections}
    except Exception as exc:
        return f"Qdrant unreachable or timed out: {exc}"

    if _COLLECTION not in existing:
        return "Knowledge base is empty — use rag_ingest to add content before searching."

    try:
        embedder = _embedder()
        query_vec = embedder.embed_query(query.strip())
    except Exception as exc:
        return f"Embedding failed: {exc}"

    try:
        result = client.query_points(
            collection_name=_COLLECTION,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        )
    except Exception as exc:
        return f"Qdrant search timed out or failed: {exc}"

    hits = [p for p in result.points if p.score >= 0.25]
    if not hits:
        return "No relevant results found in the knowledge base."

    lines = []
    for i, hit in enumerate(hits, 1):
        score = hit.score
        source = hit.payload.get("source", "unknown")
        text = hit.payload.get("text", "")
        lines.append(f"[{i}] score={score:.3f}  source={source}\n{text}")
    print('lines are', lines)

    return "\n\n".join(lines)
