"""Persistent ChromaDB Vector Store for DocuMind AI.

Encapsulates vector storage, deduplication by file hash, document-level deletion,
metadata preservation, and similarity search integration.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import settings
from src.embeddings import get_embedding_model


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class DocumentAlreadyExistsError(VectorStoreError):
    """Raised when an identical document hash has already been embedded."""
    pass


class DocumentNotFoundError(VectorStoreError):
    """Raised when attempting to delete a document that does not exist."""
    pass


class DocuMindVectorStore:
    """Manages persistent document embeddings in ChromaDB with metadata indexing."""

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[Embeddings] = None,
    ) -> None:
        """Initialize Chroma persistent client and vector store.

        Args:
            persist_directory: Directory for persistent ChromaDB storage.
            collection_name: Name of ChromaDB collection.
            embedding_model: Embedding model instance (defaults to Ollama nomic-embed-text).
        """
        self.persist_dir = persist_directory or settings.chroma_dir_path
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME

        try:
            self.embeddings = embedding_model or get_embedding_model()
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._vector_store = Chroma(
                client=self._client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to initialize ChromaDB at '{self.persist_dir}': {exc}"
            ) from exc

    @property
    def raw_collection(self):
        """Direct access to underlying Chroma collection."""
        return self._vector_store._collection

    def get_total_chunks(self) -> int:
        """Return total number of vector chunks stored in the collection."""
        try:
            return self.raw_collection.count()
        except Exception as exc:
            raise VectorStoreError(f"Error querying vector count: {exc}") from exc

    def document_exists(self, file_hash: str) -> bool:
        """Check if any chunks matching the file SHA-256 hash exist.

        Args:
            file_hash: 64-character SHA-256 hash of the uploaded document.

        Returns:
            True if document is already embedded; False otherwise.
        """
        try:
            results = self.raw_collection.get(
                where={"file_hash": file_hash},
                limit=1,
            )
            return len(results.get("ids", [])) > 0
        except Exception as exc:
            raise VectorStoreError(
                f"Error checking duplicate document with hash '{file_hash}': {exc}"
            ) from exc

    def add_documents(
        self,
        documents: List[Document],
        check_duplicates: bool = True,
    ) -> int:
        """Embed and store chunked documents into ChromaDB.

        Args:
            documents: List of chunked Document instances with metadata.
            check_duplicates: If True, checks whether the file hash is already indexed.

        Returns:
            Number of chunks successfully indexed.

        Raises:
            DocumentAlreadyExistsError: If check_duplicates is True and document exists.
            VectorStoreError: If embedding or storage fails.
        """
        if not documents:
            return 0

        # Check for duplicates using the first chunk's file_hash
        sample_hash = documents[0].metadata.get("file_hash")
        file_name = documents[0].metadata.get("file_name", "document")

        if check_duplicates and sample_hash and self.document_exists(sample_hash):
            raise DocumentAlreadyExistsError(
                f"Document '{file_name}' (hash: {sample_hash[:8]}...) is already indexed in the vector database."
            )

        # Generate or extract stable chunk IDs
        ids: List[str] = []
        for idx, doc in enumerate(documents):
            chunk_id = doc.metadata.get(
                "chunk_id",
                f"{doc.metadata.get('doc_id', 'doc')}_p{doc.metadata.get('page', 1)}_c{idx}",
            )
            ids.append(chunk_id)

        try:
            self._vector_store.add_documents(documents=documents, ids=ids)
            return len(documents)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to add document chunks to ChromaDB: {exc}"
            ) from exc

    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """Retrieve a list of all distinct documents stored in the database.

        Returns:
            List of dictionaries with document summary details:
            [{"file_name": str, "file_hash": str, "doc_id": str, "chunk_count": int, "total_pages": int}]
        """
        try:
            records = self.raw_collection.get(include=["metadatas"])
            metadatas = records.get("metadatas", []) or []

            doc_map: Dict[str, Dict[str, Any]] = {}
            for meta in metadatas:
                if not meta:
                    continue
                file_hash = meta.get("file_hash")
                if not file_hash:
                    continue

                if file_hash not in doc_map:
                    doc_map[file_hash] = {
                        "file_name": meta.get("file_name", "unknown"),
                        "file_hash": file_hash,
                        "doc_id": meta.get("doc_id", ""),
                        "chunk_count": 0,
                        "total_pages": meta.get("total_pages", meta.get("page", 1)),
                    }
                doc_map[file_hash]["chunk_count"] += 1
                current_page = meta.get("page", 1)
                if current_page > doc_map[file_hash]["total_pages"]:
                    doc_map[file_hash]["total_pages"] = current_page

            return list(doc_map.values())
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to retrieve indexed documents list: {exc}"
            ) from exc

    def delete_document(self, file_hash: str) -> int:
        """Delete all chunks belonging to a specific document hash.

        Args:
            file_hash: SHA-256 hash of document to remove.

        Returns:
            Number of chunks removed.

        Raises:
            DocumentNotFoundError: If no chunks match the given hash.
            VectorStoreError: If deletion fails.
        """
        try:
            matching = self.raw_collection.get(where={"file_hash": file_hash})
            ids_to_delete = matching.get("ids", [])

            if not ids_to_delete:
                raise DocumentNotFoundError(
                    f"No indexed document found matching hash '{file_hash}'."
                )

            self.raw_collection.delete(where={"file_hash": file_hash})
            return len(ids_to_delete)
        except DocumentNotFoundError:
            raise
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete document with hash '{file_hash}': {exc}"
            ) from exc

    def clear_all(self) -> None:
        """Clear all stored vectors and reset collection."""
        try:
            all_records = self.raw_collection.get()
            all_ids = all_records.get("ids", [])
            if all_ids:
                self.raw_collection.delete(ids=all_ids)
        except Exception as exc:
            raise VectorStoreError(f"Failed to clear vector store: {exc}") from exc

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Document]:
        """Execute similarity search against stored document vectors.

        Args:
            query: User's question or retrieval prompt.
            k: Number of nearest neighbors to return (defaults to settings.RETRIEVAL_TOP_K).

        Returns:
            List of most relevant Document chunks.
        """
        top_k = k if k is not None else settings.RETRIEVAL_TOP_K
        try:
            return self._vector_store.similarity_search(query=query, k=top_k)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to execute similarity search: {exc}"
            ) from exc

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """Execute similarity search returning document chunks and distance scores.

        Args:
            query: Retrieval prompt text.
            k: Top-k count.

        Returns:
            List of (Document, score) tuples.
        """
        top_k = k if k is not None else settings.RETRIEVAL_TOP_K
        try:
            return self._vector_store.similarity_search_with_score(query=query, k=top_k)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to execute similarity search with scores: {exc}"
            ) from exc

    def as_retriever(self, top_k: Optional[int] = None):
        """Create a standard LangChain retriever interface.

        Args:
            top_k: Number of chunks to retrieve.

        Returns:
            VectorStoreRetriever configured with search_kwargs={'k': top_k}.
        """
        k_val = top_k if top_k is not None else settings.RETRIEVAL_TOP_K
        return self._vector_store.as_retriever(search_kwargs={"k": k_val})
