"""Utility functions for DocuMind AI.

Provides file validation, hashing for deduplication, safe filename handling,
and size formatting utilities.
"""

import hashlib
import re
from pathlib import Path
from typing import BinaryIO, Union


def compute_sha256(data: Union[Path, str, bytes, BinaryIO]) -> str:
    """Compute SHA-256 hash from a file path, raw bytes, or file-like buffer.

    Args:
        data: Path to file, raw bytes, or open binary stream.

    Returns:
        Hexadecimal SHA-256 string representation.
    """
    hasher = hashlib.sha256()

    if isinstance(data, (str, Path)):
        file_path = Path(data)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
    elif isinstance(data, bytes):
        hasher.update(data)
    elif hasattr(data, "read"):
        pos = data.tell() if hasattr(data, "tell") else None
        for chunk in iter(lambda: data.read(65536), b""):
            hasher.update(chunk)
        if pos is not None and hasattr(data, "seek"):
            data.seek(pos)
    else:
        raise ValueError(f"Unsupported data type for hashing: {type(data)}")

    return hasher.hexdigest()


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal and illegal filesystem characters.

    Args:
        filename: Original filename submitted by client/user.

    Returns:
        Cleaned, safe basename string.
    """
    # Extract only the base name in case a full path was provided
    base = Path(filename).name

    # Replace forbidden Windows and Unix filename characters: < > : " / \ | ? *
    clean = re.sub(r'[<>:"/\\|?*]', "_", base)

    # Remove non-printable or control characters
    clean = re.sub(r"[\x00-\x1f\x7f]", "", clean)

    # Disallow leading periods or double dots that could cause path traversal
    clean = clean.lstrip(".").strip()

    if not clean:
        clean = "unnamed_document.pdf"

    if not clean.lower().endswith(".pdf"):
        clean = f"{clean}.pdf"

    return clean


def format_file_size(size_bytes: int) -> str:
    """Convert byte count into human-readable string (B, KB, MB).

    Args:
        size_bytes: Size in bytes.

    Returns:
        Formatted string like '1.45 MB' or '340 KB'.
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def validate_pdf_file(
    file_path: Union[str, Path],
    max_size_mb: int = 50,
) -> tuple[bool, str]:
    """Validate that a file exists, has a .pdf extension, and is within size limits.

    Args:
        file_path: Path to target file.
        max_size_mb: Maximum allowed file size in MB.

    Returns:
        Tuple of (is_valid: bool, error_message: str).
    """
    path = Path(file_path)

    if not path.exists():
        return False, f"File does not exist: {path.name}"

    if not path.is_file():
        return False, f"Target is not a regular file: {path.name}"

    if path.suffix.lower() != ".pdf":
        return False, f"Invalid file format: '{path.suffix}'. Only PDF documents are supported."

    file_size_bytes = path.stat().st_size
    if file_size_bytes == 0:
        return False, f"The uploaded PDF '{path.name}' is completely empty (0 bytes)."

    max_bytes = max_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        return False, (
            f"File '{path.name}' ({format_file_size(file_size_bytes)}) exceeds the "
            f"maximum allowed limit of {max_size_mb} MB."
        )

    return True, ""
