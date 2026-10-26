"""DocuMind AI — Intelligent Document Assistant using RAG.

A professional, local enterprise-grade Streamlit application for multi-document
PDF question answering with conversational RAG, vector deduplication, source citations,
and prompt injection defense.
"""

from pathlib import Path
from typing import List, Optional
import streamlit as st

from src.config import settings
from src.embeddings import check_ollama_status, get_embedding_model
from src.document_loader import (
    load_pdf,
    EmptyDocumentError,
    ScannedDocumentError,
    CorruptedDocumentError,
)
from src.text_splitter import split_documents
from src.vector_store import (
    DocuMindVectorStore,
    DocumentAlreadyExistsError,
    VectorStoreError,
)
from src.retriever import DocuMindRetriever
from src.rag_chain import DocuMindRAGChain, RAGTimeoutError, RAGGenerationError
from src.chat_history import (
    ChatHistoryManager,
    ChatMessage,
    ConversationalRAGChain,
)
from src.utils import compute_sha256, sanitize_filename, format_file_size


# =====================================================================
# Application Configuration & Custom Styling
# =====================================================================

st.set_page_config(
    page_title="DocuMind AI — Intelligent Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
/* Modern Dark Theme Palettes with Blue/Purple Accents */
:root {
    --bg-primary: #0f172a;
    --bg-card: #1e293b;
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --border-color: #334155;
}

/* Header Styling */
.hero-header {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 6px;
    margin-bottom: 0px;
}

/* Metric Chips */
.metric-chip {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 14px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.88rem;
    color: #e2e8f0;
    margin-right: 8px;
    margin-top: 8px;
}
.metric-chip strong {
    color: #38bdf8;
}

/* Source Citation Card */
.citation-card {
    background: #1e293b;
    border-left: 3px solid #8b5cf6;
    border-radius: 6px;
    padding: 10px 14px;
    margin-top: 8px;
    margin-bottom: 8px;
    font-size: 0.88rem;
}
.citation-title {
    font-weight: 600;
    color: #a78bfa;
    display: flex;
    align-items: center;
    gap: 6px;
}
.citation-snippet {
    color: #cbd5e1;
    font-style: italic;
    margin-top: 4px;
    padding-left: 8px;
    border-left: 2px solid #475569;
}

/* Empty State Card */
.empty-state {
    text-align: center;
    padding: 40px 20px;
    border: 2px dashed #334155;
    border-radius: 12px;
    color: #64748b;
    margin-top: 20px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================================
# Resource Caching & Initialization
# =====================================================================

@st.cache_resource(show_spinner=False)
def initialize_vector_store() -> DocuMindVectorStore:
    """Initialize singleton persistent ChromaDB vector store."""
    return DocuMindVectorStore()


def get_application_components(top_k: int):
    """Retrieve or construct retriever and conversational RAG chain."""
    vector_store = initialize_vector_store()
    retriever = DocuMindRetriever(vector_store=vector_store, default_top_k=top_k)
    rag_chain = DocuMindRAGChain(retriever=retriever)

    # Initialize chat history manager from session state
    if "chat_history_data" not in st.session_state:
        st.session_state.chat_history_data = []

    history_manager = ChatHistoryManager.from_dict_list(
        st.session_state.chat_history_data
    )
    conv_chain = ConversationalRAGChain(
        rag_chain=rag_chain,
        chat_history=history_manager,
    )
    return vector_store, retriever, conv_chain, history_manager


# =====================================================================
# Sidebar: Logo, Uploads, Processing, and Management
# =====================================================================

with st.sidebar:
    st.markdown("### 🧠 DocuMind AI")
    st.caption("Intelligent Document Assistant using Local RAG")

    # 1. Health Status Pre-flight
    ollama_ready, ollama_msg = check_ollama_status(
        required_models=[settings.OLLAMA_LLM_MODEL, settings.OLLAMA_EMBED_MODEL]
    )
    if ollama_ready:
        st.success("● Ollama Online (llama3.2 + nomic)", icon="✅")
    else:
        st.error(f"● Ollama Warning: {ollama_msg}", icon="⚠️")

    st.divider()

    # 2. Retrieval Settings
    with st.expander("⚙️ Retrieval Settings", expanded=False):
        top_k = st.slider(
            "Top-K Chunks to Retrieve",
            min_value=1,
            max_value=10,
            value=settings.RETRIEVAL_TOP_K,
            help="Number of most semantically relevant document chunks passed to the LLM.",
        )
        chunk_size = st.number_input(
            "Chunk Size (characters)",
            min_value=200,
            max_value=2500,
            value=settings.CHUNK_SIZE,
            step=100,
        )
        chunk_overlap = st.number_input(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=500,
            value=settings.CHUNK_OVERLAP,
            step=50,
        )

    # Instantiate core components
    vector_store, retriever, conv_chain, history_manager = get_application_components(
        top_k=top_k
    )

    # 3. File Upload Component
    st.markdown("#### 📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple text-based PDF documents (max 50 MB each).",
    )

    if uploaded_files:
        st.caption(f"📁 {len(uploaded_files)} file(s) selected for processing:")
        for uf in uploaded_files:
            st.text(f"• {uf.name} ({format_file_size(uf.size)})")

        if st.button("⚡ Process Documents", type="primary", use_container_width=True):
            if not ollama_ready:
                st.error("Cannot process documents: Ollama service or models are unavailable.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                success_count = 0
                duplicate_count = 0

                upload_dir = settings.upload_dir_path

                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing ({i+1}/{len(uploaded_files)}): {file.name}...")
                    file_bytes = file.read()
                    file_hash = compute_sha256(file_bytes)
                    clean_name = sanitize_filename(file.name)

                    # Deduplication check
                    if vector_store.document_exists(file_hash):
                        st.info(f"ℹ️ '{clean_name}' is already indexed in the database (skipped re-embedding).")
                        duplicate_count += 1
                        progress_bar.progress((i + 1) / len(uploaded_files))
                        continue

                    # Save to safe temporary upload path
                    temp_file_path = upload_dir / f"{file_hash[:12]}_{clean_name}"
                    try:
                        temp_file_path.write_bytes(file_bytes)

                        # Pipeline: Load -> Chunk -> Embed & Store
                        docs = load_pdf(
                            file_path=temp_file_path,
                            file_hash=file_hash,
                            max_size_mb=settings.MAX_FILE_SIZE_MB,
                        )
                        chunks = split_documents(
                            documents=docs,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                        )
                        vector_store.add_documents(chunks, check_duplicates=False)
                        success_count += 1
                    except EmptyDocumentError:
                        st.error(f"❌ '{clean_name}' contains no readable text pages.")
                    except ScannedDocumentError as exc:
                        st.warning(f"⚠️ '{clean_name}': Scanned PDF detected. OCR is required, which is outside current scope.")
                    except CorruptedDocumentError as exc:
                        st.error(f"❌ '{clean_name}' is corrupt or invalid: {exc}")
                    except Exception as exc:
                        st.error(f"❌ Failed to process '{clean_name}': {exc}")
                    finally:
                        if temp_file_path.exists():
                            temp_file_path.unlink()

                    progress_bar.progress((i + 1) / len(uploaded_files))

                status_text.empty()
                progress_bar.empty()

                if success_count > 0:
                    st.success(f"Successfully processed and indexed {success_count} new document(s)!")
                    st.rerun()
                elif duplicate_count > 0 and success_count == 0:
                    st.info("All selected documents were already indexed.")

    st.divider()

    # 4. Document Management & Inventory
    st.markdown("#### 📚 Knowledge Base")
    try:
        indexed_docs = vector_store.get_indexed_documents()
    except Exception:
        indexed_docs = []

    if indexed_docs:
        st.caption(f"Currently managing **{len(indexed_docs)}** document(s):")
        for doc in indexed_docs:
            col_info, col_del = st.columns([4, 1])
            with col_info:
                st.markdown(f"**{doc['file_name']}**")
                st.caption(f"Pages: {doc.get('total_pages', 1)} | Chunks: {doc['chunk_count']}")
            with col_del:
                if st.button("🗑️", key=f"del_{doc['file_hash']}", help=f"Delete {doc['file_name']}"):
                    vector_store.delete_document(doc["file_hash"])
                    st.toast(f"Removed '{doc['file_name']}' from knowledge base.", icon="🗑️")
                    st.rerun()

        if st.button("Clear All Documents", use_container_width=True):
            vector_store.clear_all()
            st.toast("Cleared all documents from vector store.", icon="🧹")
            st.rerun()
    else:
        st.info("No documents indexed yet. Upload a PDF above to begin.")

    st.divider()

    # 5. Conversation Controls
    if st.button("🧹 Clear Chat History", use_container_width=True):
        history_manager.clear()
        st.session_state.chat_history_data = []
        st.rerun()


# =====================================================================
# Main Section: Hero Banner & Conversational Interface
# =====================================================================

# Header Card
st.markdown(
    f"""
    <div class="hero-header">
        <h1 class="hero-title">DocuMind AI</h1>
        <p class="hero-subtitle">Intelligent, grounded document assistant powered by local RAG</p>
        <div>
            <span class="metric-chip">LLM: <strong>{settings.OLLAMA_LLM_MODEL}</strong></span>
            <span class="metric-chip">Embeddings: <strong>{settings.OLLAMA_EMBED_MODEL}</strong></span>
            <span class="metric-chip">Indexed Documents: <strong>{len(indexed_docs)}</strong></span>
            <span class="metric-chip">Vector Chunks: <strong>{vector_store.get_total_chunks()}</strong></span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display historical chat messages
chat_messages = history_manager.get_messages()

if not chat_messages:
    # Empty State & Example Questions
    st.markdown(
        """
        <div class="empty-state">
            <h3>💬 Welcome to DocuMind AI</h3>
            <p>Upload your PDF documents via the sidebar and ask questions to receive accurate, context-grounded answers with source citations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if indexed_docs:
        st.markdown("##### 💡 Suggested Questions")
        example_cols = st.columns(3)
        sample_questions = [
            "What are the main topics and objectives discussed?",
            "What key policies, dates, or numbers are specified?",
            "Summarize the most important guidelines or rules.",
        ]
        for col, eq in zip(example_cols, sample_questions):
            with col:
                if st.button(eq, use_container_width=True):
                    st.session_state["pending_input"] = eq
                    st.rerun()

# Render chat turns
for msg in chat_messages:
    with st.chat_message(msg.role):
        st.markdown(msg.content)
        # Display collapsible source citations for assistant messages
        if msg.role == "assistant" and msg.sources:
            with st.expander(f"📑 Sources & Evidence ({len(msg.sources)} references)"):
                for cit in msg.sources:
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-title">
                                📄 {cit.file_name} — Page {cit.page}
                            </div>
                            <div class="citation-snippet">
                                "{cit.snippet}"
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Handle pending input from example question buttons
user_input = st.chat_input("Ask a question about your documents...")
if "pending_input" in st.session_state and st.session_state["pending_input"]:
    user_input = st.session_state.pop("pending_input")

# Handle Question Submission
if user_input:
    # Display user turn
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant Response Generation
    with st.chat_message("assistant"):
        if not ollama_ready:
            error_response = (
                "⚠️ Ollama is offline or missing required models. "
                "Please verify your local Ollama server is running."
            )
            st.error(error_response)
            history_manager.add_user_message(user_input)
            history_manager.add_assistant_message(error_response)
            st.session_state.chat_history_data = history_manager.to_dict_list()
        elif vector_store.get_total_chunks() == 0:
            notice = (
                "No documents are currently indexed in the knowledge base. "
                "Please upload and process at least one PDF document using the sidebar."
            )
            st.warning(notice)
            history_manager.add_user_message(user_input)
            history_manager.add_assistant_message(notice)
            st.session_state.chat_history_data = history_manager.to_dict_list()
        else:
            with st.spinner("Analyzing documents and synthesizing answer..."):
                try:
                    response = conv_chain.ask(question=user_input, top_k=top_k)

                    st.markdown(response.answer)

                    if response.sources:
                        with st.expander(
                            f"📑 Sources & Evidence ({len(response.sources)} references)",
                            expanded=True,
                        ):
                            for cit in response.sources:
                                st.markdown(
                                    f"""
                                    <div class="citation-card">
                                        <div class="citation-title">
                                            📄 {cit.file_name} — Page {cit.page}
                                        </div>
                                        <div class="citation-snippet">
                                            "{cit.snippet}"
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Save updated history in Streamlit session state
                    st.session_state.chat_history_data = history_manager.to_dict_list()
                except RAGTimeoutError as exc:
                    st.error(f"⏱️ Generation timed out: {exc}")
                except Exception as exc:
                    st.error(f"❌ An error occurred during answer generation: {exc}")
