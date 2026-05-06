"""
test_mongo_user_store.py — Unit tests for tools/mongo_user_store.py

All MongoDB calls are mocked via patch("tools.mongo_user_store._get_collection").
"""

import pytest
from unittest.mock import MagicMock, patch
from pymongo.errors import DuplicateKeyError

from tools.mongo_user_store import create_user, get_user_by_email


# ── create_user ───────────────────────────────────────────────────────────────

class TestCreateUser:
    def test_insert_one_called_with_all_four_fields(self):
        mock_col = MagicMock()
        with patch("tools.mongo_user_store._get_collection", return_value=mock_col):
            create_user("uid-1", "alice@example.com", "alice", "hashed-pw")
        inserted = mock_col.insert_one.call_args[0][0]
        assert inserted["user_id"] == "uid-1"
        assert inserted["email"] == "alice@example.com"
        assert inserted["username"] == "alice"
        assert inserted["hashed_password"] == "hashed-pw"

    def test_insert_one_called_exactly_once(self):
        mock_col = MagicMock()
        with patch("tools.mongo_user_store._get_collection", return_value=mock_col):
            create_user("uid-2", "bob@example.com", "bob", "hash")
        mock_col.insert_one.assert_called_once()

    def test_duplicate_key_error_propagates(self):
        mock_col = MagicMock()
        mock_col.insert_one.side_effect = DuplicateKeyError("duplicate key error")
        with patch("tools.mongo_user_store._get_collection", return_value=mock_col):
            with pytest.raises(DuplicateKeyError):
                create_user("uid-3", "dup@example.com", "bob", "hash")


# ── get_user_by_email ─────────────────────────────────────────────────────────

class TestGetUserByEmail:
    def test_returns_user_dict_when_found(self):
        user = {"user_id": "uid-1", "email": "alice@example.com", "username": "alice"}
        mock_col = MagicMock()
        mock_col.find_one.return_value = user
        with patch("tools.mongo_user_store._get_collection", return_value=mock_col):
            result = get_user_by_email("alice@example.com")
        assert result == user

    def test_returns_none_when_not_found(self):
        mock_col = MagicMock()
        mock_col.find_one.return_value = None
        with patch("tools.mongo_user_store._get_collection", return_value=mock_col):
            result = get_user_by_email("missing@example.com")
        assert result is None

    def test_find_one_filters_by_email(self):
        mock_col = MagicMock()
        mock_col.find_one.return_value = None
        with patch("tools.mongo_user_store._get_collection", return_value=mock_col):
            get_user_by_email("test@example.com")
        filter_arg = mock_col.find_one.call_args[0][0]
        assert filter_arg == {"email": "test@example.com"}

    def test_projection_excludes_mongo_id(self):
        mock_col = MagicMock()
        mock_col.find_one.return_value = None
        with patch("tools.mongo_user_store._get_collection", return_value=mock_col):
            get_user_by_email("test@example.com")
        projection = mock_col.find_one.call_args[0][1]
        assert projection == {"_id": 0}
