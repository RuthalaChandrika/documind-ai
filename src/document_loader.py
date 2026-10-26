"""PDF Document Loader for DocuMind AI.

Loads text from PDF documents using PyPDFLoader, preserves and standardizes
metadata (1-based page numbers, document ID, file hash), and performs
strict quality checks for empty, corrupt, or scanned image-only PDFs.
"""

from pathlib import Path
from typing import List, Optional, Union
import pypdf.errors
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.utils import compute_sha256, sanitize_filename, validate_pdf_file


class DocumentProcessingError(Exception):
    """Base exception for all document processing errors."""
    pass


class CorruptedDocumentError(DocumentProcessingError):
    """Raised when a PDF file is malformed, encrypted, or corrupted."""
    pass


class EmptyDocumentError(DocumentProcessingError):
    """Raised when a PDF contains no pages or zero extractable characters."""
    pass


class ScannedDocumentError(DocumentProcessingError):
    """Raised when a PDF appears to be a scanned image without a readable text layer."""
    pass


def load_pdf(
    file_path: Union[str, Path],
    doc_id: Optional[str] = None,
    file_hash: Optional[str] = None,
    max_size_mb: int = 50,
) -> List[Document]:
    """Load a PDF document, extract text, enrich metadata, and run quality checks.

    Args:
        file_path: Path to the target PDF file on disk.
        doc_id: Unique identifier for the document. If omitted, computed from hash.
        file_hash: Pre-calculated SHA-256 hash. If omitted, computed from file.
        max_size_mb: Maximum allowed file size in MB.

    Returns:
        List of LangChain Document instances with normalized 1-based page metadata.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file fails validation (e.g., non-PDF or exceeds size).
        CorruptedDocumentError: If PDF is corrupt, encrypted, or unreadable.
        EmptyDocumentError: If PDF has no pages or zero text.
        ScannedDocumentError: If PDF text layer is empty/insufficient (scanned).
    """
    path = Path(file_path).resolve()

    # Pre-validation (existence, extension, size)
    is_valid, error_msg = validate_pdf_file(path, max_size_mb=max_size_mb)
    if not is_valid:
        if not path.exists():
            raise FileNotFoundError(error_msg)
        raise ValueError(error_msg)

    # Compute SHA-256 hash if not provided
    calculated_hash = file_hash or compute_sha256(path)
    document_id = doc_id or calculated_hash[:16]
    clean_filename = sanitize_filename(path.name)

    # Load with PyPDFLoader
    try:
        loader = PyPDFLoader(str(path))
        raw_documents = loader.load()
    except (
        pypdf.errors.PyPdfError,
        pypdf.errors.PdfStreamError,
        pypdf.errors.PdfReadError,
        pypdf.errors.EmptyFileError,
    ) as exc:
        raise CorruptedDocumentError(
            f"Failed to read '{clean_filename}'. The PDF is corrupt or invalid: {exc}"
        ) from exc
    except Exception as exc:
        raise CorruptedDocumentError(
            f"An unexpected error occurred while parsing '{clean_filename}': {exc}"
        ) from exc

    if not raw_documents:
        raise EmptyDocumentError(
            f"The PDF '{clean_filename}' contains no readable pages."
        )

    # Aggregate extracted text to detect empty or scanned documents
    total_pages = len(raw_documents)
    total_text_length = 0
    meaningful_text_count = 0

    enriched_documents: List[Document] = []

    for idx, doc in enumerate(raw_documents):
        content = (doc.page_content or "").strip()
        total_text_length += len(content)

        # Count alphanumeric characters to differentiate noise/artifacts from text
        alnum_chars = sum(1 for c in content if c.isalnum())
        meaningful_text_count += alnum_chars

        # Normalize page number to 1-based
        raw_page = doc.metadata.get("page", idx)
        page_number = int(raw_page) + 1 if isinstance(raw_page, int) else idx + 1

        doc.metadata = {
            "source": clean_filename,
            "file_name": clean_filename,
            "doc_id": document_id,
            "file_hash": calculated_hash,
            "page": page_number,
            "total_pages": total_pages,
        }
        doc.page_content = content
        enriched_documents.append(doc)

    # Quality Check: Empty or whitespace-only
    if total_text_length == 0:
        raise EmptyDocumentError(
            f"Document '{clean_filename}' contains {total_pages} page(s) but has zero text content."
        )

    # Quality Check: Scanned PDF detection
    # If the document has pages but fewer than 20 meaningful alphanumeric characters across all pages,
    # it is almost certainly a scanned document or image without an OCR layer.
    if meaningful_text_count < 20:
        raise ScannedDocumentError(
            f"Document '{clean_filename}' appears to be a scanned or image-only PDF with "
            f"insufficient extractable text ({meaningful_text_count} characters found). "
            f"Scanned PDFs require Optical Character Recognition (OCR), which is outside the current scope."
        )

    return enriched_documents
