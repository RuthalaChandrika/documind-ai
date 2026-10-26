"""Embeddings module for DocuMind AI.

Manages Ollama embedding model instantiation (nomic-embed-text) and provides
pre-flight health checking for local Ollama availability and model readiness.
"""

from typing import List, Optional, Tuple
import httpx
from langchain_ollama import OllamaEmbeddings

from src.config import settings


class OllamaServiceError(Exception):
    """Raised when the local Ollama server cannot be reached."""
    pass


class OllamaModelNotFoundError(Exception):
    """Raised when a required Ollama model is not installed locally."""
    pass


def check_ollama_status(
    base_url: Optional[str] = None,
    required_models: Optional[List[str]] = None,
    timeout_seconds: float = 3.0,
) -> Tuple[bool, str]:
    """Check if the local Ollama daemon is reachable and has required models installed.

    Args:
        base_url: Ollama API endpoint (defaults to settings.OLLAMA_BASE_URL).
        required_models: List of model names to check (e.g., ['llama3.2', 'nomic-embed-text']).
        timeout_seconds: Request timeout in seconds.

    Returns:
        Tuple of (is_ready: bool, status_message: str).
    """
    url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
    tags_endpoint = f"{url}/api/tags"

    # 1. Connection check
    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.get(tags_endpoint)
            if response.status_code != 200:
                return (
                    False,
                    f"Ollama server responded with unexpected status HTTP {response.status_code}.",
                )
            data = response.json()
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return (
            False,
            f"Cannot connect to Ollama server at '{url}'. "
            "Please ensure Ollama is installed and running (`ollama serve`).",
        )
    except Exception as exc:
        return (
            False,
            f"Error communicating with Ollama service: {exc}",
        )

    # 2. Model verification
    if required_models:
        installed_models = [
            m.get("name", "").lower() for m in data.get("models", [])
        ]
        missing_models: List[str] = []

        for req in required_models:
            req_clean = req.lower()
            # Handle tag variations (e.g., 'llama3.2' vs 'llama3.2:latest')
            has_model = any(
                installed == req_clean
                or installed.startswith(f"{req_clean}:")
                or req_clean.startswith(f"{installed}:")
                for installed in installed_models
            )
            if not has_model:
                missing_models.append(req)

        if missing_models:
            pull_cmds = "\n".join(f"  ollama pull {m}" for m in missing_models)
            return (
                False,
                f"Missing required Ollama model(s): {', '.join(missing_models)}.\n"
                f"Please download them by running:\n{pull_cmds}",
            )

    return True, "Ollama service is online and all required models are available."


def get_embedding_model(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OllamaEmbeddings:
    """Instantiate and return the OllamaEmbeddings client.

    Args:
        model_name: Embedding model name (defaults to settings.OLLAMA_EMBED_MODEL).
        base_url: Ollama server endpoint (defaults to settings.OLLAMA_BASE_URL).

    Returns:
        Configured OllamaEmbeddings instance.
    """
    selected_model = model_name or settings.OLLAMA_EMBED_MODEL
    selected_url = base_url or settings.OLLAMA_BASE_URL

    return OllamaEmbeddings(
        model=selected_model,
        base_url=selected_url,
    )
