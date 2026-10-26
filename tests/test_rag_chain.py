"""Unit and integration tests for RAG Answer Generation Chain."""

from pathlib import Path
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.vector_store import DocuMindVectorStore
from src.retriever import DocuMindRetriever
from src.rag_chain import DocuMindRAGChain, RAGResponse


@pytest.fixture
def mock_retriever(tmp_path: Path) -> DocuMindRetriever:
    """Fixture with a mock-populated retriever."""
    store = DocuMindVectorStore(
        persist_directory=tmp_path / "rag_chroma_test",
        collection_name="rag_test",
        embedding_model=FakeEmbeddings(size=768),
    )

    docs = [
        Document(
            page_content="The enterprise annual budget allocation for cloud computing is $1.2M.",
            metadata={
                "file_name": "fiscal_report_2024.pdf",
                "file_hash": "hash_fr",
                "doc_id": "doc_fr",
                "page": 15,
                "chunk_id": "doc_fr_p15_c0",
            },
        )
    ]
    store.add_documents(docs)
    return DocuMindRetriever(vector_store=store, default_top_k=1)


def test_rag_chain_empty_retrieval(tmp_path: Path):
    """Verify RAG chain handles questions when no documents exist."""
    empty_store = DocuMindVectorStore(
        persist_directory=tmp_path / "empty_rag",
        collection_name="empty_rag_col",
        embedding_model=FakeEmbeddings(size=768),
    )
    empty_retriever = DocuMindRetriever(vector_store=empty_store)
    chain = DocuMindRAGChain(retriever=empty_retriever)

    response = chain.answer_question("What is the cloud budget?")
    assert response.is_grounded is False
    assert len(response.sources) == 0
    assert "cannot find sufficient information" in response.answer.lower()


def test_rag_chain_grounded_answer(mock_retriever: DocuMindRetriever):
    """Verify RAG chain generates grounded response with citations using mock LLM."""
    mock_llm = FakeListChatModel(
        responses=[
            "The enterprise annual budget allocation for cloud computing is $1.2M [Source: fiscal_report_2024.pdf, Page 15]."
        ]
    )

    chain = DocuMindRAGChain(retriever=mock_retriever, llm=mock_llm)
    response = chain.answer_question("What is the cloud budget?")

    assert response.is_grounded is True
    assert "$1.2M" in response.answer
    assert len(response.sources) == 1
    assert response.sources[0].file_name == "fiscal_report_2024.pdf"
    assert response.sources[0].page == 15


def test_rag_chain_missing_evidence_detection(mock_retriever: DocuMindRetriever):
    """Verify that if LLM responds that information is not available, is_grounded is False."""
    mock_llm = FakeListChatModel(
        responses=[
            "I cannot find sufficient information in the provided documents to answer this question."
        ]
    )

    chain = DocuMindRAGChain(retriever=mock_retriever, llm=mock_llm)
    response = chain.answer_question("What is the remote work policy?")

    assert response.is_grounded is False
    assert len(response.sources) == 0
    assert "cannot find sufficient information" in response.answer


@pytest.mark.slow
def test_live_ollama_rag_integration(tmp_path: Path):
    """Live integration test querying local llama3.2 with real nomic-embed-text embeddings."""
    from src.embeddings import get_embedding_model

    store = DocuMindVectorStore(
        persist_directory=tmp_path / "live_chroma",
        collection_name="live_test",
        embedding_model=get_embedding_model(),
    )

    docs = [
        Document(
            page_content="Project Alpha deadline is strictly scheduled for November 15, 2026.",
            metadata={
                "file_name": "project_alpha_plan.pdf",
                "file_hash": "hash_pa",
                "doc_id": "doc_pa",
                "page": 4,
                "chunk_id": "doc_pa_p4_c0",
            },
        )
    ]
    store.add_documents(docs)
    retriever = DocuMindRetriever(vector_store=store, default_top_k=1)
    chain = DocuMindRAGChain(retriever=retriever)

    response = chain.answer_question("What is the deadline for Project Alpha?")

    assert response.is_grounded is True
    assert "november 15" in response.answer.lower()
    assert len(response.sources) >= 1
    assert response.sources[0].file_name == "project_alpha_plan.pdf"
    assert response.sources[0].page == 4
