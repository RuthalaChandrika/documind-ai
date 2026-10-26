"""Unit tests for PDF loading, validation, and text chunking."""

from pathlib import Path
import pytest
import pymupdf

from src.document_loader import (
    load_pdf,
    EmptyDocumentError,
    ScannedDocumentError,
    CorruptedDocumentError,
)
from src.text_splitter import split_documents, get_text_splitter
from src.utils import compute_sha256, sanitize_filename, validate_pdf_file


@pytest.fixture
def temp_pdf_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for generating test PDFs."""
    return tmp_path


def create_sample_pdf(file_path: Path, pages_text: list[str]) -> Path:
    """Helper to create a multi-page PDF with specified text content."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text.strip():
            page.insert_text((50, 72), text)
    doc.save(str(file_path))
    doc.close()
    return file_path


def test_sanitize_filename():
    """Verify filename sanitization against path traversal and illegal characters."""
    assert sanitize_filename("safe_report.pdf") == "safe_report.pdf"
    assert sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename("my:file<test>|name.pdf") == "my_file_test__name.pdf"
    assert sanitize_filename("document_without_ext") == "document_without_ext.pdf"
    assert sanitize_filename("...").endswith(".pdf")


def test_compute_sha256(temp_pdf_dir: Path):
    """Verify SHA-256 hash calculation on files and bytes."""
    file_a = temp_pdf_dir / "file_a.txt"
    file_b = temp_pdf_dir / "file_b.txt"
    file_c = temp_pdf_dir / "file_c.txt"

    file_a.write_text("Identical content for DocuMind AI testing.", encoding="utf-8")
    file_b.write_text("Identical content for DocuMind AI testing.", encoding="utf-8")
    file_c.write_text("Different content.", encoding="utf-8")

    hash_a = compute_sha256(file_a)
    hash_b = compute_sha256(file_b)
    hash_c = compute_sha256(file_c)

    assert hash_a == hash_b
    assert hash_a != hash_c
    assert len(hash_a) == 64


def test_load_valid_pdf(temp_pdf_dir: Path):
    """Verify standard text-based PDF loading and metadata preservation."""
    pdf_path = temp_pdf_dir / "annual_policy.pdf"
    page1_text = "DocuMind AI provides retrieval-augmented generation for enterprise documents."
    page2_text = "Employees should submit leave requests at least five working days in advance."

    create_sample_pdf(pdf_path, [page1_text, page2_text])

    docs = load_pdf(pdf_path)

    assert len(docs) == 2
    # Page 1
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["total_pages"] == 2
    assert docs[0].metadata["file_name"] == "annual_policy.pdf"
    assert docs[0].metadata["source"] == "annual_policy.pdf"
    assert docs[0].metadata["doc_id"] is not None
    assert docs[0].metadata["file_hash"] is not None
    assert "retrieval-augmented generation" in docs[0].page_content

    # Page 2
    assert docs[1].metadata["page"] == 2
    assert docs[1].metadata["total_pages"] == 2
    assert "five working days" in docs[1].page_content


def test_empty_pdf_raises_error(temp_pdf_dir: Path):
    """Verify empty PDF without pages or text raises EmptyDocumentError."""
    pdf_path = temp_pdf_dir / "empty.pdf"
    create_sample_pdf(pdf_path, [""])

    with pytest.raises(EmptyDocumentError):
        load_pdf(pdf_path)


def test_scanned_pdf_raises_error(temp_pdf_dir: Path):
    """Verify PDF with negligible alphanumeric text triggers ScannedDocumentError."""
    pdf_path = temp_pdf_dir / "scanned_invoice.pdf"
    # Only 3 characters, below the 20 meaningful characters threshold
    create_sample_pdf(pdf_path, ["..."])

    with pytest.raises(ScannedDocumentError) as exc_info:
        load_pdf(pdf_path)

    assert "OCR" in str(exc_info.value)


def test_corrupted_pdf_raises_error(temp_pdf_dir: Path):
    """Verify corrupted/malformed PDF raises CorruptedDocumentError."""
    pdf_path = temp_pdf_dir / "corrupted.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 completely invalid binary stream garbage")

    with pytest.raises(CorruptedDocumentError):
        load_pdf(pdf_path)


def test_text_splitting_and_metadata_enrichment(temp_pdf_dir: Path):
    """Verify RecursiveCharacterTextSplitter generates bounded chunks with complete provenance."""
    pdf_path = temp_pdf_dir / "knowledge_base.pdf"

    long_paragraph = (
        "DocuMind AI utilizes retrieval-augmented generation to provide accurate, "
        "evidence-backed answers from enterprise document repositories. "
        "The system retrieves semantically relevant passages, contextualizes user queries, "
        "and produces strictly grounded natural-language responses with document and page citations.\n\n"
    ) * 10

    create_sample_pdf(pdf_path, [long_paragraph])
    docs = load_pdf(pdf_path)

    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 1
    for idx, chunk in enumerate(chunks):
        assert len(chunk.page_content) <= 300
        assert chunk.metadata["file_name"] == "knowledge_base.pdf"
        assert chunk.metadata["page"] == 1
        assert chunk.metadata["chunk_index"] == idx
        assert chunk.metadata["chunk_id"] == f"{docs[0].metadata['doc_id']}_p1_c{idx}"
        assert chunk.metadata["char_count"] == len(chunk.page_content)
