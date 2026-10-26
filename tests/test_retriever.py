"""Unit tests for semantic retriever and citation extraction."""

from pathlib import Path
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings

from src.vector_store import DocuMindVectorStore
from src.retriever import (
    DocuMindRetriever,
    extract_citations,
    format_retrieved_context,
)


@pytest.fixture
def populated_retriever(tmp_path: Path) -> DocuMindRetriever:
    """Fixture returning a retriever pre-loaded with synthetic documents."""
    store = DocuMindVectorStore(
        persist_directory=tmp_path / "chroma_retriever_test",
        collection_name="retriever_test",
        embedding_model=FakeEmbeddings(size=768),
    )

    docs = [
        Document(
            page_content="Standard work hours are 9:00 AM to 5:00 PM EST from Monday through Friday.",
            metadata={
                "file_name": "handbook.pdf",
                "file_hash": "hash_hb",
                "doc_id": "doc_hb",
                "page": 3,
                "chunk_id": "doc_hb_p3_c0",
            },
        ),
        Document(
            page_content="Employees are eligible for 20 days of paid annual leave each fiscal year.",
            metadata={
                "file_name": "leave_policy.pdf",
                "file_hash": "hash_lp",
                "doc_id": "doc_lp",
                "page": 7,
                "chunk_id": "doc_lp_p7_c1",
            },
        ),
    ]
    store.add_documents(docs)
    return DocuMindRetriever(vector_store=store, default_top_k=2)


def test_empty_retriever(tmp_path: Path):
    """Verify empty vector store returns an empty list without raising errors."""
    empty_store = DocuMindVectorStore(
        persist_directory=tmp_path / "empty_chroma",
        collection_name="empty_test",
        embedding_model=FakeEmbeddings(size=768),
    )
    retriever = DocuMindRetriever(vector_store=empty_store)

    results = retriever.retrieve("What are the hours?")
    assert results == []

    docs, citations = retriever.retrieve_with_sources("What are the hours?")
    assert docs == []
    assert citations == []


def test_retrieve_with_sources(populated_retriever: DocuMindRetriever):
    """Verify retrieve_with_sources returns chunks and properly populated citations."""
    docs, citations = populated_retriever.retrieve_with_sources(
        "What are the working hours?", top_k=2
    )

    assert len(docs) > 0
    assert len(citations) == len(docs)

    first_cit = citations[0]
    assert first_cit.file_name in ["handbook.pdf", "leave_policy.pdf"]
    assert first_cit.page in [3, 7]
    assert len(first_cit.snippet) > 0


def test_format_retrieved_context():
    """Verify context formatting wraps chunks in security demarcation boundaries."""
    docs = [
        Document(
            page_content="Server maintenance occurs on Saturday at midnight.",
            metadata={"file_name": "ops.pdf", "page": 4, "chunk_id": "ops_c0"},
        )
    ]

    formatted = format_retrieved_context(docs)
    assert "START OF RETRIEVED UNTRUSTED CONTEXT" in formatted
    assert "ops.pdf | Page 4" in formatted
    assert "Saturday at midnight" in formatted
    assert "END OF RETRIEVED UNTRUSTED CONTEXT" in formatted


def test_extract_citations_deduplication():
    """Verify identical chunk IDs are deduplicated in citation list."""
    docs = [
        Document(
            page_content="Content snippet A.",
            metadata={"file_name": "a.pdf", "page": 1, "chunk_id": "a_1"},
        ),
        Document(
            page_content="Content snippet A duplicated.",
            metadata={"file_name": "a.pdf", "page": 1, "chunk_id": "a_1"},
        ),
    ]

    citations = extract_citations(docs)
    assert len(citations) == 1
    assert citations[0].chunk_id == "a_1"
