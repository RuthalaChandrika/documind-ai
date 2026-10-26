"""Tests for Streamlit App UI Initialization using AppTest."""

import pytest
from streamlit.testing.v1 import AppTest


def test_app_initialization_no_exceptions():
    """Verify Streamlit application initializes, mounts sidebar, and renders without exceptions."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Ensure no uncaught exceptions occurred during render
    assert not at.exception
    # Verify main title and sidebar components exist
    assert "DocuMind AI" in at.sidebar.markdown[0].value
