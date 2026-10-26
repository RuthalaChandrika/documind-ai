"""Unit tests for Conversational RAG, Chat History, and Query Contextualization."""

from pathlib import Path
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.vector_store import DocuMindVectorStore
from src.retriever import DocuMindRetriever, SourceCitation
from src.rag_chain import DocuMindRAGChain
from src.chat_history import (
    ChatHistoryManager,
    ChatMessage,
    contextualize_query,
    ConversationalRAGChain,
)


def test_chat_history_manager_crud():
    """Verify adding messages, formatting history, clearing, and serialization."""
    manager = ChatHistoryManager()
    assert len(manager.get_messages()) == 0

    manager.add_user_message("What is the annual leave quota?")
    manager.add_assistant_message(
        "Employees receive 20 days.",
        sources=[SourceCitation(file_name="policy.pdf", page=2, snippet="20 days")],
    )

    messages = manager.get_messages()
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert len(messages[1].sources) == 1

    formatted = manager.format_history_for_prompt()
    assert "User: What is the annual leave quota?" in formatted
    assert "Assistant: Employees receive 20 days." in formatted

    # Test serialization round-trip
    serialized = manager.to_dict_list()
    restored = ChatHistoryManager.from_dict_list(serialized)
    assert len(restored.get_messages()) == 2
    assert restored.get_messages()[1].content == "Employees receive 20 days."

    # Test clearing history
    manager.clear()
    assert len(manager.get_messages()) == 0


def test_contextualize_query_with_empty_history():
    """Verify empty chat history leaves question unchanged without calling LLM."""
    manager = ChatHistoryManager()
    mock_llm = FakeListChatModel(responses=["Unexpected reformulation"])

    query = "What is the health insurance policy?"
    reformulated = contextualize_query(query, manager, mock_llm)
    assert reformulated == query


def test_contextualize_query_with_prior_turn():
    """Verify follow-up question is reformulated into a standalone retrieval query."""
    manager = ChatHistoryManager()
    manager.add_user_message("What is the company leave policy?")
    manager.add_assistant_message("Employees can take up to 20 days of annual leave.")

    mock_llm = FakeListChatModel(
        responses=["How many days in advance should I apply for annual leave?"]
    )

    query = "How many days in advance should I apply?"
    standalone = contextualize_query(query, manager, mock_llm)
    assert standalone == "How many days in advance should I apply for annual leave?"


def test_conversational_rag_chain_multiturn(tmp_path: Path):
    """Verify end-to-end multi-turn conversation flow updates session history."""
    store = DocuMindVectorStore(
        persist_directory=tmp_path / "conv_chroma",
        collection_name="conv_test",
        embedding_model=FakeEmbeddings(size=768),
    )
    store.add_documents(
        [
            Document(
                page_content="Annual leave requests must be submitted at least 5 working days in advance.",
                metadata={
                    "file_name": "leave.pdf",
                    "file_hash": "h_leave",
                    "doc_id": "d_leave",
                    "page": 4,
                    "chunk_id": "d_leave_p4_c0",
                },
            )
        ]
    )

    retriever = DocuMindRetriever(vector_store=store, default_top_k=1)
    rag_chain = DocuMindRAGChain(
        retriever=retriever,
        llm=FakeListChatModel(
            responses=[
                "Employees get 20 days [Source: leave.pdf, Page 4].",
                "You must apply at least 5 working days in advance [Source: leave.pdf, Page 4].",
            ]
        ),
    )

    conv_chain = ConversationalRAGChain(
        rag_chain=rag_chain,
        contextualize_llm=FakeListChatModel(
            responses=["How many days in advance should I apply for annual leave?"]
        ),
    )

    # Turn 1
    resp1 = conv_chain.ask("What is the leave quota?")
    assert "20 days" in resp1.answer
    assert len(conv_chain.chat_history.get_messages()) == 2

    # Turn 2: Follow-up question
    resp2 = conv_chain.ask("How many days in advance should I apply?")
    assert "5 working days" in resp2.answer
    assert resp2.query == "How many days in advance should I apply?"
    assert len(conv_chain.chat_history.get_messages()) == 4
