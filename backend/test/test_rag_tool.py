"""
test_rag_tool.py — Unit tests for the rag_tool function in mcp_server.py

embed_text and similarity_search are patched at mcp_server's import location.
"""

import pytest
from unittest.mock import patch

from mcp_server import rag_tool


_FAKE_EMBEDDING = [0.1, 0.2, 0.3]


class TestRagTool:
    def test_no_results_returns_no_documents_message(self):
        with patch("mcp_server.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("mcp_server.similarity_search", return_value=[]):
            result = rag_tool("progressive overload")
        assert "No relevant documents found" in result

    def test_single_result_contains_source(self):
        hits = [("Some training advice text.", 0.91, "strength_manual.pdf")]
        with patch("mcp_server.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("mcp_server.similarity_search", return_value=hits):
            result = rag_tool("bench press technique")
        assert "strength_manual.pdf" in result

    def test_single_result_contains_text(self):
        hits = [("Some training advice text.", 0.91, "strength_manual.pdf")]
        with patch("mcp_server.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("mcp_server.similarity_search", return_value=hits):
            result = rag_tool("query")
        assert "Some training advice text." in result

    def test_similarity_score_formatted_to_2_decimal_places(self):
        hits = [("text", 0.91234, "doc.pdf")]
        with patch("mcp_server.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("mcp_server.similarity_search", return_value=hits):
            result = rag_tool("query")
        assert "0.91" in result
        assert "0.91234" not in result

    def test_multiple_results_separated_by_divider(self):
        hits = [
            ("text one", 0.95, "doc1.pdf"),
            ("text two", 0.80, "doc2.pdf"),
        ]
        with patch("mcp_server.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("mcp_server.similarity_search", return_value=hits):
            result = rag_tool("query")
        assert "---" in result

    def test_multiple_results_all_sources_present(self):
        hits = [
            ("text one", 0.95, "doc1.pdf"),
            ("text two", 0.80, "doc2.pdf"),
        ]
        with patch("mcp_server.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("mcp_server.similarity_search", return_value=hits):
            result = rag_tool("query")
        assert "doc1.pdf" in result
        assert "doc2.pdf" in result

    def test_result_contains_source_prefix(self):
        hits = [("text", 0.9, "manual.pdf")]
        with patch("mcp_server.embed_text", return_value=_FAKE_EMBEDDING), \
             patch("mcp_server.similarity_search", return_value=hits):
            result = rag_tool("query")
        assert "[Source:" in result
