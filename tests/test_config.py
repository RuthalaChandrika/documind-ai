"""Tests for configuration settings."""

from pathlib import Path
from src.config import settings, PROJECT_ROOT


def test_settings_loaded():
    """Verify default and environment settings load properly."""
    assert settings.OLLAMA_LLM_MODEL == "llama3.2"
    assert settings.OLLAMA_EMBED_MODEL == "nomic-embed-text"
    assert settings.CHUNK_SIZE > 0
    assert settings.CHUNK_OVERLAP >= 0
    assert settings.RETRIEVAL_TOP_K > 0
    assert isinstance(settings.chroma_dir_path, Path)
    assert isinstance(settings.upload_dir_path, Path)
    assert settings.chroma_dir_path.exists()
    assert settings.upload_dir_path.exists()
