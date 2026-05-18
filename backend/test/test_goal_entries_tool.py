"""
test_goal_entries_tool.py — Unit tests for goal_entries_tool in mcp_server.py

get_goal and get_goal_entries are patched at mcp_server's import location.
"""

import pytest
from unittest.mock import patch

from mcp_server import goal_entries_tool


_GOAL = {
    "goal_id": "g1",
    "user_id": "user-1",
    "exercise_name": "Squat",
    "target_reps": 1,
    "target_weight_kg": 140.0,
}


class TestGoalEntriesTool:
    def test_no_goal_returns_correct_message(self):
        with patch("mcp_server.get_goal", return_value=None):
            result = goal_entries_tool("user-1")
        assert "No active goal found" in result
        assert "user-1" in result

    def test_goal_exists_no_entries_describes_target(self):
        with patch("mcp_server.get_goal", return_value=_GOAL), \
             patch("mcp_server.get_goal_entries", return_value=[]):
            result = goal_entries_tool("user-1")
        assert "Squat" in result
        assert "140.0" in result

    def test_goal_and_entries_shows_history_header(self):
        entries = [{
            "date": "2025-01-15T10:00:00",
            "sets": [{"reps": 3, "weight_kg": 130.0}],
        }]
        with patch("mcp_server.get_goal", return_value=_GOAL), \
             patch("mcp_server.get_goal_entries", return_value=entries):
            result = goal_entries_tool("user-1")
        assert "history" in result.lower()

    def test_entry_data_appears_in_output(self):
        entries = [{
            "date": "2025-01-15T10:00:00",
            "sets": [{"reps": 3, "weight_kg": 130.0}],
        }]
        with patch("mcp_server.get_goal", return_value=_GOAL), \
             patch("mcp_server.get_goal_entries", return_value=entries):
            result = goal_entries_tool("user-1")
        assert "130.0" in result

    def test_entry_with_rpe_includes_rpe(self):
        entries = [{
            "date": "2025-01-15T10:00:00",
            "sets": [{"reps": 3, "weight_kg": 130.0, "rpe": 9}],
        }]
        with patch("mcp_server.get_goal", return_value=_GOAL), \
             patch("mcp_server.get_goal_entries", return_value=entries):
            result = goal_entries_tool("user-1")
        assert "RPE" in result

    def test_entry_without_rpe_omits_rpe(self):
        entries = [{
            "date": "2025-01-15T10:00:00",
            "sets": [{"reps": 3, "weight_kg": 130.0}],
        }]
        with patch("mcp_server.get_goal", return_value=_GOAL), \
             patch("mcp_server.get_goal_entries", return_value=entries):
            result = goal_entries_tool("user-1")
        assert "RPE" not in result

    def test_entry_with_notes_includes_notes(self):
        entries = [{
            "date": "2025-01-15T10:00:00",
            "sets": [{"reps": 3, "weight_kg": 130.0}],
            "notes": "belt felt tight",
        }]
        with patch("mcp_server.get_goal", return_value=_GOAL), \
             patch("mcp_server.get_goal_entries", return_value=entries):
            result = goal_entries_tool("user-1")
        assert "belt felt tight" in result

    def test_entry_without_notes_omits_notes(self):
        entries = [{
            "date": "2025-01-15T10:00:00",
            "sets": [{"reps": 3, "weight_kg": 130.0}],
        }]
        with patch("mcp_server.get_goal", return_value=_GOAL), \
             patch("mcp_server.get_goal_entries", return_value=entries):
            result = goal_entries_tool("user-1")
        assert "belt felt tight" not in result
