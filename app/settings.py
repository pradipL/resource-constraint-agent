import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # LLM provider
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai").lower())
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    openai_embed_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))

    # Tavily & Qdrant
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "knowledge_base"))
    max_llm_calls: int = field(default_factory=lambda: int(os.getenv("MAX_LLM_CALLS", "10")))
    max_completion_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_COMPLETION_TOKENS", "16384")))
    max_token_limit: int = field(default_factory=lambda: int(os.getenv("MAX_TOKEN_LIMIT", "10")))
    model_max_output_tokens: int = field(default_factory=lambda: int(os.getenv("MODEL_MAX_OUTPUT_TOKENS", "4096")))
    summary_token_buffer: int = field(default_factory=lambda: int(os.getenv("SUMMARY_TOKEN_BUFFER", "100")))

    def __post_init__(self):
        missing = [
            name for name, val in [
                ("OPENAI_API_KEY", self.openai_api_key),
                ("TAVILY_API_KEY", self.tavily_api_key),
            ]
            if not val
        ]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


settings = Settings()
