"""Configuration settings for DocuMind AI.

Manages application parameters, Ollama endpoints, model identifiers,
text chunking configurations, and persistent storage directories.
"""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root path (directory containing this repository)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """DocuMind AI application settings loaded from environment or defaults."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama Service
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama server",
    )
    OLLAMA_LLM_MODEL: str = Field(
        default="llama3.2",
        description="Local LLM model tag for generative responses",
    )
    OLLAMA_EMBED_MODEL: str = Field(
        default="nomic-embed-text",
        description="Local embedding model tag for vector representations",
    )
    LLM_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for LLM deterministic output",
    )
    LLM_TIMEOUT_SECONDS: float = Field(
        default=60.0,
        gt=0.0,
        description="Request timeout for Ollama invocations",
    )

    # Text Splitting
    CHUNK_SIZE: int = Field(
        default=1000,
        gt=0,
        description="Target character length for document text chunks",
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        ge=0,
        description="Character overlap between consecutive document chunks",
    )

    # ChromaDB Vector Store
    CHROMA_PERSIST_DIRECTORY: str = Field(
        default="chroma_db",
        description="Relative or absolute path for ChromaDB persistence",
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default="documind_documents",
        description="Collection name in ChromaDB for document vectors",
    )

    # Retrieval
    RETRIEVAL_TOP_K: int = Field(
        default=4,
        gt=0,
        description="Default number of top relevant chunks to retrieve",
    )

    # File Management
    UPLOAD_DIRECTORY: str = Field(
        default="data/uploads",
        description="Directory where uploaded PDF documents are stored",
    )
    MAX_FILE_SIZE_MB: int = Field(
        default=50,
        gt=0,
        description="Maximum permissible upload size per file in megabytes",
    )

    @property
    def chroma_dir_path(self) -> Path:
        """Return resolved Path to the ChromaDB directory."""
        path = Path(self.CHROMA_PERSIST_DIRECTORY)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def upload_dir_path(self) -> Path:
        """Return resolved Path to the upload storage directory."""
        path = Path(self.UPLOAD_DIRECTORY)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path


# Singleton configuration instance
settings = Settings()
