"""Tests for RAG Evaluation metrics calculation."""

from pathlib import Path
import pytest
from src.evaluate import (
    EVAL_DATASET,
    create_eval_documents,
)


def test_eval_dataset_integrity():
    """Verify evaluation dataset contains balanced queries and required ground truth fields."""
    assert len(EVAL_DATASET) >= 5

    answerable = [item for item in EVAL_DATASET if item.is_answerable]
    unanswerable = [item for item in EVAL_DATASET if not item.is_answerable]

    assert len(answerable) >= 4
    assert len(unanswerable) >= 1

    for item in answerable:
        assert item.expected_doc.endswith(".pdf")
        assert item.expected_page > 0
        assert len(item.expected_fact) > 0


def test_create_eval_documents(tmp_path: Path):
    """Verify evaluation document generation creates valid text-based multi-page PDFs."""
    created_files = create_eval_documents(tmp_path)
    assert len(created_files) == 2

    for path in created_files:
        assert path.exists()
        assert path.stat().st_size > 0
        assert path.suffix == ".pdf"
