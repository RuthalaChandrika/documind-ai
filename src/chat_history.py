"""Conversational RAG and Chat History Management for DocuMind AI.

Maintains multi-turn conversation sessions, reformulates follow-up queries
into standalone retrieval prompts, and prevents cross-session history leakage.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.retriever import SourceCitation
from src.rag_chain import DocuMindRAGChain, RAGResponse


class ChatMessage(BaseModel):
    """Represents a single conversational turn in the chat interface."""

    role: Literal["user", "assistant"] = Field(description="Role of the message author")
    content: str = Field(description="Body of the message")
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="Source citations supporting assistant answers",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )


CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question which might reference context "
    "in the chat history, formulate a concise standalone question that can be understood "
    "without the chat history. Do NOT answer the question, do NOT add introductory text "
    "or explanations, just reformulate it if needed, and otherwise return the original question as is."
)

CONTEXTUALIZE_USER_PROMPT = """Chat History:
{chat_history}

Latest Question: {question}

Standalone Question:"""


def get_contextualize_prompt() -> ChatPromptTemplate:
    """Return configured ChatPromptTemplate for query reformulation."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            ("human", CONTEXTUALIZE_USER_PROMPT),
        ]
    )


class ChatHistoryManager:
    """Manages an isolated conversation history per session."""

    def __init__(self, initial_messages: Optional[List[ChatMessage]] = None) -> None:
        """Initialize chat history with an optional list of pre-existing messages."""
        self._messages: List[ChatMessage] = initial_messages or []

    def add_user_message(self, content: str) -> ChatMessage:
        """Record a user query message."""
        msg = ChatMessage(role="user", content=content.strip())
        self._messages.append(msg)
        return msg

    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List[SourceCitation]] = None,
    ) -> ChatMessage:
        """Record an assistant response message with attached citations."""
        msg = ChatMessage(
            role="assistant",
            content=content.strip(),
            sources=sources or [],
        )
        self._messages.append(msg)
        return msg

    def get_messages(self) -> List[ChatMessage]:
        """Return a copy of all messages in chronological order."""
        return list(self._messages)

    def get_recent_history(self, max_turns: int = 6) -> List[ChatMessage]:
        """Return the most recent message turns."""
        return self._messages[-max_turns:]

    def format_history_for_prompt(self, max_turns: int = 6) -> str:
        """Format recent conversation turns into a clean dialogue string."""
        recent = self.get_recent_history(max_turns=max_turns)
        if not recent:
            return ""

        lines: List[str] = []
        for msg in recent:
            role_label = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{role_label}: {msg.content}")

        return "\n".join(lines)

    def clear(self) -> None:
        """Erase all messages in this session history."""
        self._messages.clear()

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Serialize message history to a list of dictionaries for Streamlit session storage."""
        return [msg.model_dump() for msg in self._messages]

    @classmethod
    def from_dict_list(cls, data: List[Dict[str, Any]]) -> "ChatHistoryManager":
        """Instantiate ChatHistoryManager from serialized dictionary list."""
        messages = [ChatMessage(**item) for item in data]
        return cls(initial_messages=messages)


def contextualize_query(
    question: str,
    chat_history: ChatHistoryManager,
    llm: BaseChatModel,
    max_history_turns: int = 6,
) -> str:
    """Reformulate a follow-up question into an independent standalone search query.

    If chat history is empty or reformulation fails, returns the original query.

    Args:
        question: User's latest input question.
        chat_history: Session ChatHistoryManager instance.
        llm: Language model used for contextualization.
        max_history_turns: Number of recent messages to provide as context.

    Returns:
        Standalone query string optimized for semantic vector retrieval.
    """
    clean_q = question.strip()
    history_str = chat_history.format_history_for_prompt(max_turns=max_history_turns)

    if not history_str:
        return clean_q

    prompt_template = get_contextualize_prompt()
    messages = prompt_template.format_messages(
        chat_history=history_str,
        question=clean_q,
    )

    try:
        response = llm.invoke(messages)
        standalone = response.content.strip()
        # Clean quotes if model wrapped response in quotes
        if standalone.startswith('"') and standalone.endswith('"'):
            standalone = standalone[1:-1].strip()
        return standalone if standalone else clean_q
    except Exception:
        # Fall back gracefully to original query on any network or timeout failure
        return clean_q


class ConversationalRAGChain:
    """Coordinates multi-turn conversational retrieval and question answering."""

    def __init__(
        self,
        rag_chain: DocuMindRAGChain,
        chat_history: Optional[ChatHistoryManager] = None,
        contextualize_llm: Optional[BaseChatModel] = None,
    ) -> None:
        """Initialize conversational chain.

        Args:
            rag_chain: Core DocuMindRAGChain for grounded answers.
            chat_history: Session chat history manager.
            contextualize_llm: LLM used for query reformulation (defaults to rag_chain.llm).
        """
        self.rag_chain = rag_chain
        self.chat_history = chat_history or ChatHistoryManager()
        self.contextualize_llm = contextualize_llm or rag_chain.llm

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """Process a conversational turn: contextualize query, retrieve, answer, and store history.

        Args:
            question: Follow-up or initial user question.
            top_k: Number of retrieved chunks.

        Returns:
            RAGResponse grounded in retrieved documents.
        """
        clean_question = question.strip()
        if not clean_question:
            return RAGResponse(
                query=question,
                answer="Please enter a valid question to search your documents.",
                sources=[],
                is_grounded=False,
                retrieved_documents=[],
            )

        # 1. Contextualize query if prior turns exist
        standalone_query = contextualize_query(
            question=clean_question,
            chat_history=self.chat_history,
            llm=self.contextualize_llm,
        )

        # 2. Answer using the standalone query for retrieval
        response = self.rag_chain.answer_question(
            question=standalone_query,
            top_k=top_k,
        )

        # Restore original query on response object for UI clarity
        response.query = clean_question

        # 3. Append turn to session history
        self.chat_history.add_user_message(clean_question)
        self.chat_history.add_assistant_message(
            content=response.answer,
            sources=response.sources,
        )

        return response
