"""Tests for ChromaDB Vector Store, Deduplication, and Embeddings."""

from pathlib import Path
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings

from src.vector_store import (
    DocuMindVectorStore,
    DocumentAlreadyExistsError,
    DocumentNotFoundError,
)
from src.embeddings import check_ollama_status, get_embedding_model


@pytest.fixture
def fake_embedding_model():
    """Deterministic fake embedding model for fast unit testing."""
    return FakeEmbeddings(size=768)


@pytest.fixture
def temp_vector_store(tmp_path: Path, fake_embedding_model):
    """Fixture providing a fresh isolated DocuMindVectorStore instance in tmp_path."""
    store = DocuMindVectorStore(
        persist_directory=tmp_path / "chroma_test",
        collection_name="test_collection",
        embedding_model=fake_embedding_model,
    )
    return store


def test_ollama_status_check():
    """Verify pre-flight health check against live local Ollama service."""
    status, msg = check_ollama_status(required_models=["llama3.2", "nomic-embed-text"])
    assert status is True
    assert "online" in msg.lower()


def test_add_documents_and_count(temp_vector_store: DocuMindVectorStore):
    """Verify adding chunked documents increments vector count and stores metadata."""
    docs = [
        Document(
            page_content="DocuMind AI provides accurate document-grounded question answering.",
            metadata={
                "file_name": "handbook.pdf",
                "file_hash": "hash_handbook_001",
                "doc_id": "doc_001",
                "page": 1,
                "total_pages": 1,
                "chunk_id": "doc_001_p1_c0",
            },
        )
    ]

    added = temp_vector_store.add_documents(docs)
    assert added == 1
    assert temp_vector_store.get_total_chunks() == 1
    assert temp_vector_store.document_exists("hash_handbook_001") is True
    assert temp_vector_store.document_exists("non_existent_hash") is False


def test_duplicate_upload_rejection(temp_vector_store: DocuMindVectorStore):
    """Verify attempting to add the same document hash raises DocumentAlreadyExistsError."""
    docs = [
        Document(
            page_content="Policy guideline on vacation leave and sick days.",
            metadata={
                "file_name": "leave_policy.pdf",
                "file_hash": "hash_leave_policy_123",
                "doc_id": "doc_leave",
                "page": 1,
                "total_pages": 1,
                "chunk_id": "doc_leave_p1_c0",
            },
        )
    ]

    temp_vector_store.add_documents(docs)

    # Second attempt with same hash must be rejected to prevent unnecessary re-embedding
    with pytest.raises(DocumentAlreadyExistsError):
        temp_vector_store.add_documents(docs, check_duplicates=True)


def test_get_indexed_documents(temp_vector_store: DocuMindVectorStore):
    """Verify aggregated listing of distinct indexed documents and page counts."""
    docs_a = [
        Document(
            page_content="First chunk of doc A.",
            metadata={
                "file_name": "doc_a.pdf",
                "file_hash": "hash_a",
                "doc_id": "id_a",
                "page": 1,
                "total_pages": 2,
                "chunk_id": "id_a_p1_c0",
            },
        ),
        Document(
            page_content="Second chunk of doc A.",
            metadata={
                "file_name": "doc_a.pdf",
                "file_hash": "hash_a",
                "doc_id": "id_a",
                "page": 2,
                "total_pages": 2,
                "chunk_id": "id_a_p2_c1",
            },
        ),
    ]

    docs_b = [
        Document(
            page_content="Only chunk of doc B.",
            metadata={
                "file_name": "doc_b.pdf",
                "file_hash": "hash_b",
                "doc_id": "id_b",
                "page": 1,
                "total_pages": 1,
                "chunk_id": "id_b_p1_c0",
            },
        )
    ]

    temp_vector_store.add_documents(docs_a)
    temp_vector_store.add_documents(docs_b)

    indexed = temp_vector_store.get_indexed_documents()
    assert len(indexed) == 2

    hashes = {doc["file_hash"] for doc in indexed}
    assert "hash_a" in hashes
    assert "hash_b" in hashes

    doc_a_info = next(d for d in indexed if d["file_hash"] == "hash_a")
    assert doc_a_info["chunk_count"] == 2
    assert doc_a_info["total_pages"] == 2


def test_delete_document(temp_vector_store: DocuMindVectorStore):
    """Verify document-level deletion removes all its chunks from the collection."""
    docs = [
        Document(
            page_content="Chunk 1 of removable doc.",
            metadata={
                "file_name": "remove_me.pdf",
                "file_hash": "hash_remove_me",
                "doc_id": "id_rem",
                "page": 1,
                "chunk_id": "id_rem_p1_c0",
            },
        ),
        Document(
            page_content="Chunk 2 of removable doc.",
            metadata={
                "file_name": "remove_me.pdf",
                "file_hash": "hash_remove_me",
                "doc_id": "id_rem",
                "page": 2,
                "chunk_id": "id_rem_p2_c1",
            },
        ),
    ]

    temp_vector_store.add_documents(docs)
    assert temp_vector_store.get_total_chunks() == 2

    deleted_count = temp_vector_store.delete_document("hash_remove_me")
    assert deleted_count == 2
    assert temp_vector_store.get_total_chunks() == 0
    assert temp_vector_store.document_exists("hash_remove_me") is False

    # Deleting non-existent document raises DocumentNotFoundError
    with pytest.raises(DocumentNotFoundError):
        temp_vector_store.delete_document("hash_remove_me")


def test_similarity_search(temp_vector_store: DocuMindVectorStore):
    """Verify similarity search returns expected chunks with provenance metadata."""
    docs = [
        Document(
            page_content="The annual health insurance deductible is $500 per family member.",
            metadata={
                "file_name": "benefits.pdf",
                "file_hash": "hash_benefits",
                "doc_id": "doc_ben",
                "page": 14,
                "chunk_id": "doc_ben_p14_c0",
            },
        )
    ]
    temp_vector_store.add_documents(docs)

    results = temp_vector_store.similarity_search("What is the health deductible?", k=1)
    assert len(results) == 1
    assert "deductible is $500" in results[0].page_content
    assert results[0].metadata["file_name"] == "benefits.pdf"
    assert results[0].metadata["page"] == 14
