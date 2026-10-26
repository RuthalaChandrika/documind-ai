"""Text chunking module for DocuMind AI.

Splits loaded PDF documents into semantic chunks using RecursiveCharacterTextSplitter,
preserving document provenance (file name, 1-based page, doc_id, file_hash)
and assigning deterministic chunk identifiers for precise source citations.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def get_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> RecursiveCharacterTextSplitter:
    """Create a configured RecursiveCharacterTextSplitter instance.

    Args:
        chunk_size: Target maximum characters per chunk (defaults to settings.CHUNK_SIZE).
        chunk_overlap: Overlapping character count (defaults to settings.CHUNK_OVERLAP).

    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP

    if overlap >= size:
        raise ValueError(
            f"chunk_overlap ({overlap}) must be strictly less than chunk_size ({size})."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )


def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """Split a list of LangChain Document objects into chunked Document instances.

    Preserves existing metadata from each document and enriches each chunk with:
    - 'chunk_id': Unique deterministic identifier '{doc_id}_p{page}_c{index}'
    - 'chunk_index': Zero-based index within the resulting chunk stream
    - 'char_count': Character count of the chunk's text

    Args:
        documents: List of input Document objects (typically output of load_pdf).
        chunk_size: Target maximum character length per chunk.
        chunk_overlap: Overlap length between adjacent chunks.

    Returns:
        List of chunked Document objects ready for vector embedding and storage.
    """
    if not documents:
        return []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raw_chunks = splitter.split_documents(documents)

    chunked_documents: List[Document] = []
    chunk_counter = 0

    for chunk in raw_chunks:
        text = chunk.page_content.strip()
        if not text:
            continue

        doc_id = chunk.metadata.get("doc_id", "doc")
        page = chunk.metadata.get("page", 1)

        # Unique deterministic chunk id
        chunk_id = f"{doc_id}_p{page}_c{chunk_counter}"

        enriched_metadata = dict(chunk.metadata)
        enriched_metadata.update(
            {
                "chunk_id": chunk_id,
                "chunk_index": chunk_counter,
                "char_count": len(text),
            }
        )

        chunked_documents.append(
            Document(
                page_content=text,
                metadata=enriched_metadata,
            )
        )
        chunk_counter += 1

    return chunked_documents
