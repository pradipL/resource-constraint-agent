"""
Run this script to ingest app/tasks/knowledge_base.txt into Qdrant.

    python ingest_kb.py

Requires Qdrant and Ollama (nomic-embed-text) to be running.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

KB_PATH = Path(__file__).parent / "app" / "tasks" / "knowledge_base2.txt"


def main() -> None:
    if not KB_PATH.exists():
        print(f"Error: {KB_PATH} not found", file=sys.stderr)
        sys.exit(1)

    content = KB_PATH.read_text(encoding="utf-8")

    from app.tools.rag_tool import ingest_text

    result = ingest_text(content, source="knowledge_base")
    print(result)


if __name__ == "__main__":
    main()
