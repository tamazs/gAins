"""
test_session_history_tool.py — Unit tests for agents/tools/session_history_tool.py

The tool is called via its LangChain .invoke() interface so that the full
tool wrapping is exercised. get_recent_sessions is patched at the tool's
import location so the mock is active when the function body runs.
"""

import pytest
from unittest.mock import patch

from agents.tools.session_history_tool import session_history_tool


def _invoke(user_id: str) -> str:
    return session_history_tool.invoke({"user_id": user_id})


class TestSessionHistoryTool:
    def test_no_sessions_returns_correct_message(self):
        with patch("agents.tools.session_history_tool.get_recent_sessions", return_value=[]):
            result = _invoke("user-1")
        assert "No previous sessions found" in result
        assert "user-1" in result

    def test_single_session_contains_date(self):
        sessions = [{
            "date": "2025-01-15",
            "exercises": [{"name": "Bench Press", "muscle_group": "chest",
                           "sets": [{"reps": 5, "weight_kg": 100.0}]}],
        }]
        with patch("agents.tools.session_history_tool.get_recent_sessions", return_value=sessions):
            result = _invoke("user-1")
        assert "2025-01-15" in result

    def test_single_session_contains_exercise_name(self):
        sessions = [{
            "date": "2025-01-15",
            "exercises": [{"name": "Bench Press", "muscle_group": "chest",
                           "sets": [{"reps": 5, "weight_kg": 100.0}]}],
        }]
        with patch("agents.tools.session_history_tool.get_recent_sessions", return_value=sessions):
            result = _invoke("user-1")
        assert "Bench Press" in result

    def test_single_session_contains_set_summary(self):
        sessions = [{
            "date": "2025-01-15",
            "exercises": [{"name": "Bench Press", "muscle_group": "chest",
                           "sets": [{"reps": 5, "weight_kg": 100.0}]}],
        }]
        with patch("agents.tools.session_history_tool.get_recent_sessions", return_value=sessions):
            result = _invoke("user-1")
        assert "5 reps @ 100.0kg" in result

    def test_multiple_sessions_both_appear(self):
        sessions = [
            {"date": "2025-01-15", "exercises": [
                {"name": "Squat", "muscle_group": "quads",
                 "sets": [{"reps": 5, "weight_kg": 120.0}]}]},
            {"date": "2025-01-10", "exercises": [
                {"name": "Deadlift", "muscle_group": "back",
                 "sets": [{"reps": 3, "weight_kg": 140.0}]}]},
        ]
        with patch("agents.tools.session_history_tool.get_recent_sessions", return_value=sessions):
            result = _invoke("user-1")
        assert "Squat" in result
        assert "Deadlift" in result

    def test_session_with_notes_includes_notes(self):
        sessions = [{
            "date": "2025-01-15",
            "notes": "felt very tired",
            "exercises": [{"name": "Squat", "muscle_group": "quads",
                           "sets": [{"reps": 5, "weight_kg": 120.0}]}],
        }]
        with patch("agents.tools.session_history_tool.get_recent_sessions", return_value=sessions):
            result = _invoke("user-1")
        assert "felt very tired" in result

    def test_session_without_notes_omits_notes_label(self):
        sessions = [{
            "date": "2025-01-15",
            "exercises": [{"name": "Squat", "muscle_group": "quads",
                           "sets": [{"reps": 5, "weight_kg": 120.0}]}],
        }]
        with patch("agents.tools.session_history_tool.get_recent_sessions", return_value=sessions):
            result = _invoke("user-1")
        assert "Notes:" not in result

    def test_set_with_rpe_includes_rpe_in_output(self):
        sessions = [{
            "date": "2025-01-15",
            "exercises": [{"name": "Squat", "muscle_group": "quads",
                           "sets": [{"reps": 5, "weight_kg": 120.0, "rpe": 8}]}],
        }]
        with patch("agents.tools.session_history_tool.get_recent_sessions", return_value=sessions):
            result = _invoke("user-1")
        assert "RPE" in result

    def test_set_without_rpe_omits_rpe_from_output(self):
        sessions = [{
            "date": "2025-01-15",
            "exercises": [{"name": "Squat", "muscle_group": "quads",
                           "sets": [{"reps": 5, "weight_kg": 120.0}]}],
        }]
        with patch("agents.tools.session_history_tool.get_recent_sessions", return_value=sessions):
            result = _invoke("user-1")
        assert "RPE" not in result
