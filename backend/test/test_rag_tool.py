"""
test_rag_tool.py — Unit tests for agents/tools/rag_tool.py

embed_text and similarity_search are patched at the tool's import location.
"""

import pytest
from unittest.mock import patch

from agents.tools.rag_tool import rag_tool


def _invoke(query: str) -> str:
    return rag_tool.invoke({"query": query})


_FAKE_EMBEDDING = [0.1, 0.2, 0.3]


class TestRagTool:
    def test_no_results_returns_no_documents_message(self):
        with patch("agents.tools.rag_tool.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("agents.tools.rag_tool.similarity_search", return_value=[]):
            result = _invoke("progressive overload")
        assert "No relevant documents found" in result

    def test_single_result_contains_source(self):
        hits = [("Some training advice text.", 0.91, "strength_manual.pdf")]
        with patch("agents.tools.rag_tool.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("agents.tools.rag_tool.similarity_search", return_value=hits):
            result = _invoke("bench press technique")
        assert "strength_manual.pdf" in result

    def test_single_result_contains_text(self):
        hits = [("Some training advice text.", 0.91, "strength_manual.pdf")]
        with patch("agents.tools.rag_tool.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("agents.tools.rag_tool.similarity_search", return_value=hits):
            result = _invoke("query")
        assert "Some training advice text." in result

    def test_similarity_score_formatted_to_2_decimal_places(self):
        hits = [("text", 0.91234, "doc.pdf")]
        with patch("agents.tools.rag_tool.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("agents.tools.rag_tool.similarity_search", return_value=hits):
            result = _invoke("query")
        assert "0.91" in result
        assert "0.91234" not in result

    def test_multiple_results_separated_by_divider(self):
        hits = [
            ("text one", 0.95, "doc1.pdf"),
            ("text two", 0.80, "doc2.pdf"),
        ]
        with patch("agents.tools.rag_tool.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("agents.tools.rag_tool.similarity_search", return_value=hits):
            result = _invoke("query")
        assert "---" in result

    def test_multiple_results_all_sources_present(self):
        hits = [
            ("text one", 0.95, "doc1.pdf"),
            ("text two", 0.80, "doc2.pdf"),
        ]
        with patch("agents.tools.rag_tool.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("agents.tools.rag_tool.similarity_search", return_value=hits):
            result = _invoke("query")
        assert "doc1.pdf" in result
        assert "doc2.pdf" in result

    def test_result_contains_source_prefix(self):
        hits = [("text", 0.9, "manual.pdf")]
        with patch("agents.tools.rag_tool.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("agents.tools.rag_tool.similarity_search", return_value=hits):
            result = _invoke("query")
        assert "[Source:" in result
