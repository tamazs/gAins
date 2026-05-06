"""
test_mongo_session_store.py — Unit tests for tools/mongo_session_store.py

All MongoDB calls are mocked at the _get_*_collection() function level.
The reset_mongo_singletons autouse fixture (conftest.py) ensures the lazy-init
singletons are cleared before each test.
"""

import pytest
from unittest.mock import MagicMock, patch
from pymongo import DESCENDING

from tools.mongo_session_store import (
    save_session,
    get_recent_sessions,
    save_goal,
    get_goal,
    save_goal_entry,
    get_goal_entries,
)


def _chain_mock(docs=None):
    """
    Return a MagicMock collection whose .find().sort().limit() chain
    returns `docs` (defaulting to an empty list).
    """
    col = MagicMock()
    col.find.return_value.sort.return_value.limit.return_value = docs or []
    return col


# ── save_session ──────────────────────────────────────────────────────────────

class TestSaveSession:
    def test_insert_one_called_with_session_id_merged(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_collection", return_value=mock_col):
            save_session("sess-1", {"user_id": "u1", "date": "2025-01-01"})
        inserted = mock_col.insert_one.call_args[0][0]
        assert inserted["session_id"] == "sess-1"
        assert inserted["user_id"] == "u1"
        assert inserted["date"] == "2025-01-01"

    def test_insert_one_called_exactly_once(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_collection", return_value=mock_col):
            save_session("sess-2", {"user_id": "u2"})
        mock_col.insert_one.assert_called_once()


# ── get_recent_sessions ───────────────────────────────────────────────────────

class TestGetRecentSessions:
    def test_returns_list_of_sessions(self):
        docs = [{"session_id": "s1"}, {"session_id": "s2"}]
        mock_col = _chain_mock(docs)
        with patch("tools.mongo_session_store._get_collection", return_value=mock_col):
            result = get_recent_sessions("u1")
        assert result == docs

    def test_empty_result_returns_empty_list(self):
        mock_col = _chain_mock([])
        with patch("tools.mongo_session_store._get_collection", return_value=mock_col):
            result = get_recent_sessions("u1")
        assert result == []

    def test_default_limit_is_5(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_collection", return_value=mock_col):
            get_recent_sessions("u1")
        mock_col.find.return_value.sort.return_value.limit.assert_called_once_with(5)

    def test_custom_limit_passed_through(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_collection", return_value=mock_col):
            get_recent_sessions("u1", limit=3)
        mock_col.find.return_value.sort.return_value.limit.assert_called_once_with(3)

    def test_sort_called_with_date_descending(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_collection", return_value=mock_col):
            get_recent_sessions("u1")
        mock_col.find.return_value.sort.assert_called_once_with("date", DESCENDING)

    def test_find_filters_by_user_id(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_collection", return_value=mock_col):
            get_recent_sessions("target-user")
        filter_arg = mock_col.find.call_args[0][0]
        assert filter_arg["user_id"] == "target-user"


# ── save_goal ─────────────────────────────────────────────────────────────────

class TestSaveGoal:
    def test_replace_one_called_with_upsert_true(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_goals_collection", return_value=mock_col):
            save_goal("goal-1", {"user_id": "u1", "exercise_name": "Squat"})
        _, kwargs = mock_col.replace_one.call_args
        assert kwargs.get("upsert") is True

    def test_filter_uses_user_id(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_goals_collection", return_value=mock_col):
            save_goal("goal-1", {"user_id": "u1", "exercise_name": "Squat"})
        filter_arg = mock_col.replace_one.call_args[0][0]
        assert filter_arg == {"user_id": "u1"}

    def test_replacement_includes_goal_id(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_goals_collection", return_value=mock_col):
            save_goal("goal-xyz", {"user_id": "u1", "exercise_name": "Squat"})
        replacement = mock_col.replace_one.call_args[0][1]
        assert replacement["goal_id"] == "goal-xyz"

    def test_replacement_includes_goal_data(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_goals_collection", return_value=mock_col):
            save_goal("g1", {"user_id": "u1", "exercise_name": "Deadlift"})
        replacement = mock_col.replace_one.call_args[0][1]
        assert replacement["exercise_name"] == "Deadlift"


# ── get_goal ──────────────────────────────────────────────────────────────────

class TestGetGoal:
    def test_returns_goal_dict_when_found(self):
        mock_col = _chain_mock()
        mock_col.find_one.return_value = {"goal_id": "g1", "user_id": "u1"}
        with patch("tools.mongo_session_store._get_goals_collection", return_value=mock_col):
            result = get_goal("u1")
        assert result == {"goal_id": "g1", "user_id": "u1"}

    def test_returns_none_when_not_found(self):
        mock_col = _chain_mock()
        mock_col.find_one.return_value = None
        with patch("tools.mongo_session_store._get_goals_collection", return_value=mock_col):
            result = get_goal("u1")
        assert result is None

    def test_projection_excludes_mongo_id(self):
        mock_col = _chain_mock()
        mock_col.find_one.return_value = {}
        with patch("tools.mongo_session_store._get_goals_collection", return_value=mock_col):
            get_goal("u1")
        projection = mock_col.find_one.call_args[0][1]
        assert projection == {"_id": 0}


# ── save_goal_entry ───────────────────────────────────────────────────────────

class TestSaveGoalEntry:
    def test_insert_one_called_with_entry_id_and_user_id(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_goal_entries_collection", return_value=mock_col):
            save_goal_entry("entry-1", "u1", {"reps": 5, "weight_kg": 100.0})
        inserted = mock_col.insert_one.call_args[0][0]
        assert inserted["entry_id"] == "entry-1"
        assert inserted["user_id"] == "u1"

    def test_insert_one_merges_entry_data(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_goal_entries_collection", return_value=mock_col):
            save_goal_entry("e1", "u1", {"exercise_name": "Squat", "reps": 3})
        inserted = mock_col.insert_one.call_args[0][0]
        assert inserted["exercise_name"] == "Squat"
        assert inserted["reps"] == 3


# ── get_goal_entries ──────────────────────────────────────────────────────────

class TestGetGoalEntries:
    def test_returns_entries_list(self):
        docs = [{"entry_id": "e1"}, {"entry_id": "e2"}]
        mock_col = _chain_mock(docs)
        with patch("tools.mongo_session_store._get_goal_entries_collection", return_value=mock_col):
            result = get_goal_entries("u1", "Squat")
        assert result == docs

    def test_empty_result_returns_empty_list(self):
        mock_col = _chain_mock([])
        with patch("tools.mongo_session_store._get_goal_entries_collection", return_value=mock_col):
            result = get_goal_entries("u1", "Squat")
        assert result == []

    def test_filters_by_user_id(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_goal_entries_collection", return_value=mock_col):
            get_goal_entries("target-user", "Squat")
        filter_arg = mock_col.find.call_args[0][0]
        assert filter_arg["user_id"] == "target-user"

    def test_filters_by_exercise_name(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_goal_entries_collection", return_value=mock_col):
            get_goal_entries("u1", "Deadlift")
        filter_arg = mock_col.find.call_args[0][0]
        assert filter_arg["exercise_name"] == "Deadlift"

    def test_sort_called_with_date_descending(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_goal_entries_collection", return_value=mock_col):
            get_goal_entries("u1", "Squat")
        mock_col.find.return_value.sort.assert_called_once_with("date", DESCENDING)

    def test_custom_limit_passed_through(self):
        mock_col = _chain_mock()
        with patch("tools.mongo_session_store._get_goal_entries_collection", return_value=mock_col):
            get_goal_entries("u1", "Squat", limit=5)
        mock_col.find.return_value.sort.return_value.limit.assert_called_once_with(5)
