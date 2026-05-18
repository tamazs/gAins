"""
test_gains_agent.py — Unit tests for agents/gains_agent.py

_SourceTracker is tested in full isolation — its on_tool_end logic is the
core extraction logic worth verifying.

GainsAgent.run is tested by replacing _run_async with an AsyncMock so we can
verify return shapes and source propagation without hitting Ollama or MCP.
"""

import pytest
from unittest.mock import AsyncMock

from agents.gains_agent import _SourceTracker, GainsAgent


# ── _SourceTracker ────────────────────────────────────────────────────────────

class TestSourceTracker:
    def test_initial_sources_list_is_empty(self):
        tracker = _SourceTracker()
        assert tracker.sources == []

    def test_no_source_lines_leaves_sources_empty(self):
        tracker = _SourceTracker()
        tracker.on_tool_end("Some output with no source lines\nJust regular text.")
        assert tracker.sources == []

    def test_single_source_line_is_extracted(self):
        tracker = _SourceTracker()
        tracker.on_tool_end("[Source: training_manual.pdf | similarity: 0.91]\nSome text")
        assert tracker.sources == ["training_manual.pdf"]

    def test_source_without_pipe_separator_is_extracted(self):
        """Source lines ending with ] directly (no pipe) should still work."""
        tracker = _SourceTracker()
        tracker.on_tool_end("[Source: doc.pdf]\nsome text")
        assert "doc.pdf" in tracker.sources

    def test_duplicate_sources_are_deduplicated(self):
        tracker = _SourceTracker()
        tracker.on_tool_end("[Source: doc.pdf | similarity: 0.91]\nText 1")
        tracker.on_tool_end("[Source: doc.pdf | similarity: 0.88]\nText 2")
        assert tracker.sources.count("doc.pdf") == 1

    def test_multiple_distinct_sources_all_collected(self):
        tracker = _SourceTracker()
        tracker.on_tool_end("[Source: doc1.pdf | similarity: 0.91]\nText 1")
        tracker.on_tool_end("[Source: doc2.pdf | similarity: 0.88]\nText 2")
        assert "doc1.pdf" in tracker.sources
        assert "doc2.pdf" in tracker.sources
        assert len(tracker.sources) == 2

    def test_non_source_lines_are_ignored(self):
        tracker = _SourceTracker()
        tracker.on_tool_end("line1\nline2\n[NotSource: something]\n[source: lowercase]")
        assert tracker.sources == []

    def test_source_name_is_stripped_of_whitespace(self):
        tracker = _SourceTracker()
        tracker.on_tool_end("[Source:   spaced_doc.pdf | similarity: 0.7]\ntext")
        assert "spaced_doc.pdf" in tracker.sources


# ── GainsAgent.run ────────────────────────────────────────────────────────────

class TestGainsAgentRun:
    def test_run_returns_tuple(self):
        agent = GainsAgent()
        agent._run_async = AsyncMock(return_value=('{"result": "ok"}', []))
        result = agent.run("test input")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_run_output_is_string(self):
        agent = GainsAgent()
        agent._run_async = AsyncMock(return_value=('{"key": "value"}', []))
        output, _ = agent.run("test input")
        assert isinstance(output, str)

    def test_run_sources_is_list(self):
        agent = GainsAgent()
        agent._run_async = AsyncMock(return_value=("result", []))
        _, sources = agent.run("test input")
        assert isinstance(sources, list)

    def test_run_output_matches_async_output(self):
        agent = GainsAgent()
        agent._run_async = AsyncMock(return_value=("expected output", []))
        output, _ = agent.run("some prompt")
        assert output == "expected output"

    def test_sources_returned_from_async(self):
        agent = GainsAgent()
        agent._run_async = AsyncMock(return_value=("result", ["science.pdf"]))
        _, sources = agent.run("test")
        assert "science.pdf" in sources

    def test_empty_sources_when_none_returned(self):
        agent = GainsAgent()
        agent._run_async = AsyncMock(return_value=("direct answer", []))
        _, sources = agent.run("prompt")
        assert sources == []
