"""
test_main.py — Integration tests for all FastAPI endpoints in main.py

All MongoDB helper functions are patched at the main.* namespace (where they
are imported into) rather than at their source modules.

main.agent is a GainsAgent instance whose _executor is a MagicMock (because
langchain_classic.agents is stubbed in conftest.py).  For each test that
exercises an AI endpoint, we use patch.object(main.agent, "run", ...) to
control the return value without touching Ollama.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from pymongo.errors import DuplicateKeyError

import main
from main import app

client = TestClient(app)

# ── Shared test data ──────────────────────────────────────────────────────────

_PAST_DATE = "2024-06-15T10:00:00Z"
_FUTURE_DATE_STR = "2099-01-01T00:00:00Z"
_FUTURE_DEADLINE = "2030-12-31"


def _session_body(**overrides):
    base = {
        "user_id": "user-1",
        "date": _PAST_DATE,
        "exercises": [
            {"name": "Bench Press", "muscle_group": "chest",
             "sets": [{"reps": 5, "weight_kg": 100.0}]}
        ],
    }
    base.update(overrides)
    return base


def _goal_body(**overrides):
    base = {
        "user_id": "user-1",
        "exercise_name": "Squat",
        "muscle_group": "quads",
        "target_weight_kg": 140.0,
        "target_reps": 1,
    }
    base.update(overrides)
    return base


def _entry_body(**overrides):
    base = {
        "user_id": "user-1",
        "date": _PAST_DATE,
        "sets": [{"reps": 3, "weight_kg": 130.0}],
    }
    base.update(overrides)
    return base


def _goal_doc(**overrides):
    base = {
        "goal_id": "goal-abc",
        "user_id": "user-1",
        "exercise_name": "Squat",
        "muscle_group": "quads",
        "target_weight_kg": 140.0,
        "target_reps": 1,
        "deadline": None,
        "notes": None,
        "created_at": "2025-01-01T00:00:00",
    }
    base.update(overrides)
    return base


_AGENT_SESSION_JSON = json.dumps({
    "overall_summary": "Good session",
    "exercise_advice": [
        {"exercise_name": "Bench Press", "recommendation": "Increase weight",
         "reasoning": "Completed all sets comfortably"}
    ],
    "recovery_flag": False,
    "sources_used": [],
})

_AGENT_GOAL_JSON = json.dumps({
    "advice": "Keep adding 2.5kg per week",
    "next_session_suggestion": "Try 135kg x3 next session",
})

_ENTRY_DOC = {
    "entry_id": "e1",
    "user_id": "user-1",
    "exercise_name": "Squat",
    "date": "2025-01-15T10:00:00",
    "sets": [{"reps": 3, "weight_kg": 130.0, "rpe": None}],
    "notes": None,
}


# ── POST /auth/register ───────────────────────────────────────────────────────

class TestRegisterEndpoint:
    def test_successful_registration_returns_200(self):
        with patch("main.create_user"):
            response = client.post("/auth/register", json={
                "email": "alice@example.com",
                "password": "password123",
                "username": "alice",
            })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["username"] == "alice"

    def test_duplicate_email_returns_409(self):
        with patch("main.create_user", side_effect=DuplicateKeyError("dup")):
            response = client.post("/auth/register", json={
                "email": "dup@example.com",
                "password": "password123",
                "username": "bob",
            })
        assert response.status_code == 409

    def test_invalid_email_format_returns_422(self):
        response = client.post("/auth/register", json={
            "email": "notanemail",
            "password": "password123",
            "username": "alice",
        })
        assert response.status_code == 422

    def test_short_password_returns_422(self):
        response = client.post("/auth/register", json={
            "email": "alice@example.com",
            "password": "short",
            "username": "alice",
        })
        assert response.status_code == 422

    def test_empty_username_returns_422(self):
        response = client.post("/auth/register", json={
            "email": "alice@example.com",
            "password": "password123",
            "username": "",
        })
        assert response.status_code == 422

    def test_response_contains_user_id(self):
        with patch("main.create_user"):
            response = client.post("/auth/register", json={
                "email": "carol@example.com",
                "password": "password123",
                "username": "carol",
            })
        assert "user_id" in response.json()


# ── POST /auth/login ──────────────────────────────────────────────────────────

class TestLoginEndpoint:
    @staticmethod
    def _user_with_password(pw: str) -> dict:
        from auth import hash_password
        return {
            "user_id": "uid-1",
            "email": "alice@example.com",
            "username": "alice",
            "hashed_password": hash_password(pw),
        }

    def test_successful_login_returns_200(self):
        user = self._user_with_password("password123")
        with patch("main.get_user_by_email", return_value=user):
            response = client.post("/auth/login", json={
                "email": "alice@example.com",
                "password": "password123",
            })
        assert response.status_code == 200
        assert "access_token" in response.json()

    def test_wrong_password_returns_401(self):
        user = self._user_with_password("correct-password")
        with patch("main.get_user_by_email", return_value=user):
            response = client.post("/auth/login", json={
                "email": "alice@example.com",
                "password": "wrong-password",
            })
        assert response.status_code == 401

    def test_nonexistent_email_returns_401(self):
        with patch("main.get_user_by_email", return_value=None):
            response = client.post("/auth/login", json={
                "email": "nobody@example.com",
                "password": "password123",
            })
        assert response.status_code == 401

    def test_email_normalized_before_db_lookup(self):
        """Login request email should be lowercased before get_user_by_email is called."""
        with patch("main.get_user_by_email", return_value=None) as mock_fn:
            client.post("/auth/login", json={
                "email": "USER@EXAMPLE.COM",
                "password": "pw",
            })
        mock_fn.assert_called_once_with("user@example.com")


# ── POST /sessions ────────────────────────────────────────────────────────────

class TestAnalyseSessionEndpoint:
    def test_successful_session_returns_200_with_advice(self):
        with patch("main.save_session"), \
             patch.object(main.agent, "run", return_value=(_AGENT_SESSION_JSON, [])):
            response = client.post("/sessions", json=_session_body())
        assert response.status_code == 200
        data = response.json()
        assert data["overall_summary"] == "Good session"
        assert isinstance(data["exercise_advice"], list)

    def test_agent_non_json_response_returns_500(self):
        with patch("main.save_session"), \
             patch.object(main.agent, "run", return_value=("not valid json", [])):
            response = client.post("/sessions", json=_session_body())
        assert response.status_code == 500

    def test_future_date_returns_422(self):
        response = client.post("/sessions", json=_session_body(date=_FUTURE_DATE_STR))
        assert response.status_code == 422

    def test_duplicate_exercises_returns_422(self):
        ex = {"name": "Bench Press", "muscle_group": "chest",
              "sets": [{"reps": 5, "weight_kg": 100.0}]}
        response = client.post("/sessions", json=_session_body(exercises=[ex, ex]))
        assert response.status_code == 422

    def test_missing_exercises_returns_422(self):
        response = client.post("/sessions", json=_session_body(exercises=[]))
        assert response.status_code == 422

    def test_save_session_called_on_success(self):
        with patch("main.save_session") as mock_save, \
             patch.object(main.agent, "run", return_value=(_AGENT_SESSION_JSON, [])):
            client.post("/sessions", json=_session_body())
        mock_save.assert_called_once()


# ── GET /sessions/{user_id} ───────────────────────────────────────────────────

class TestGetSessionsEndpoint:
    def test_returns_list_of_sessions(self):
        docs = [{"session_id": "s1"}, {"session_id": "s2"}]
        with patch("main.get_recent_sessions", return_value=docs):
            response = client.get("/sessions/user-1")
        assert response.status_code == 200
        assert len(response.json()) == 2

    def test_empty_list_returned_for_new_user(self):
        with patch("main.get_recent_sessions", return_value=[]):
            response = client.get("/sessions/user-1")
        assert response.status_code == 200
        assert response.json() == []

    def test_default_limit_param_forwarded(self):
        with patch("main.get_recent_sessions", return_value=[]) as mock_fn:
            client.get("/sessions/user-1")
        mock_fn.assert_called_once_with("user-1", limit=10)

    def test_custom_limit_query_param_forwarded(self):
        with patch("main.get_recent_sessions", return_value=[]) as mock_fn:
            client.get("/sessions/user-1?limit=3")
        mock_fn.assert_called_once_with("user-1", limit=3)


# ── POST /goals ───────────────────────────────────────────────────────────────

class TestSetGoalEndpoint:
    def test_successful_goal_creation_returns_200(self):
        with patch("main.save_goal"):
            response = client.post("/goals", json=_goal_body())
        assert response.status_code == 200
        data = response.json()
        assert data["exercise_name"] == "Squat"
        assert "goal_id" in data
        assert "created_at" in data

    def test_invalid_muscle_group_returns_422(self):
        response = client.post("/goals", json=_goal_body(muscle_group="invalid"))
        assert response.status_code == 422

    def test_past_deadline_returns_422(self):
        response = client.post("/goals", json=_goal_body(deadline="2020-01-01"))
        assert response.status_code == 422

    def test_zero_target_weight_returns_422(self):
        response = client.post("/goals", json=_goal_body(target_weight_kg=0))
        assert response.status_code == 422

    def test_zero_target_reps_returns_422(self):
        response = client.post("/goals", json=_goal_body(target_reps=0))
        assert response.status_code == 422

    def test_save_goal_called_on_success(self):
        with patch("main.save_goal") as mock_save:
            client.post("/goals", json=_goal_body())
        mock_save.assert_called_once()


# ── GET /goals/{user_id} ──────────────────────────────────────────────────────

class TestRetrieveGoalEndpoint:
    def test_existing_goal_returns_200_with_data(self):
        with patch("main.get_goal", return_value=_goal_doc()):
            response = client.get("/goals/user-1")
        assert response.status_code == 200
        data = response.json()
        assert data["exercise_name"] == "Squat"
        assert data["goal_id"] == "goal-abc"

    def test_missing_goal_returns_404(self):
        with patch("main.get_goal", return_value=None):
            response = client.get("/goals/user-1")
        assert response.status_code == 404


# ── POST /goals/entries ───────────────────────────────────────────────────────

class TestLogGoalEntryEndpoint:
    def test_successful_entry_returns_200_with_entry_id(self):
        with patch("main.get_goal", return_value=_goal_doc()), \
             patch("main.save_goal_entry"):
            response = client.post("/goals/entries", json=_entry_body())
        assert response.status_code == 200
        data = response.json()
        assert "entry_id" in data
        assert data["exercise_name"] == "Squat"

    def test_no_active_goal_returns_404(self):
        with patch("main.get_goal", return_value=None):
            response = client.post("/goals/entries", json=_entry_body())
        assert response.status_code == 404

    def test_future_date_returns_422(self):
        response = client.post("/goals/entries", json=_entry_body(date=_FUTURE_DATE_STR))
        assert response.status_code == 422

    def test_exercise_name_copied_from_goal(self):
        goal = _goal_doc(exercise_name="Deadlift")
        with patch("main.get_goal", return_value=goal), \
             patch("main.save_goal_entry"):
            response = client.post("/goals/entries", json=_entry_body())
        assert response.json()["exercise_name"] == "Deadlift"


# ── GET /goals/entries/{user_id} ──────────────────────────────────────────────

class TestGetGoalEntriesEndpoint:
    def test_returns_entries_when_goal_exists(self):
        with patch("main.get_goal", return_value=_goal_doc()), \
             patch("main.get_goal_entries", return_value=[_ENTRY_DOC]):
            response = client.get("/goals/entries/user-1")
        assert response.status_code == 200
        assert len(response.json()) == 1

    def test_returns_empty_list_when_no_entries(self):
        with patch("main.get_goal", return_value=_goal_doc()), \
             patch("main.get_goal_entries", return_value=[]):
            response = client.get("/goals/entries/user-1")
        assert response.status_code == 200
        assert response.json() == []

    def test_no_goal_returns_404(self):
        with patch("main.get_goal", return_value=None):
            response = client.get("/goals/entries/user-1")
        assert response.status_code == 404

    def test_default_limit_passed_to_store(self):
        with patch("main.get_goal", return_value=_goal_doc()), \
             patch("main.get_goal_entries", return_value=[]) as mock_fn:
            client.get("/goals/entries/user-1")
        # get_goal_entries called with user_id, exercise_name, limit
        assert mock_fn.called
        kwargs = mock_fn.call_args[1] if mock_fn.call_args[1] else {}
        args = mock_fn.call_args[0]
        # limit=20 is the default
        limit = kwargs.get("limit") or (args[2] if len(args) > 2 else None)
        assert limit == 20


# ── POST /goals/analyse ───────────────────────────────────────────────────────

class TestAnalyseGoalEndpoint:
    def test_successful_analysis_returns_200(self):
        with patch("main.get_goal", return_value=_goal_doc()), \
             patch.object(main.agent, "run", return_value=(_AGENT_GOAL_JSON, ["doc.pdf"])):
            response = client.post("/goals/analyse", json={"user_id": "user-1"})
        assert response.status_code == 200
        data = response.json()
        assert data["advice"] == "Keep adding 2.5kg per week"
        assert data["next_session_suggestion"] == "Try 135kg x3 next session"

    def test_sources_included_in_response(self):
        with patch("main.get_goal", return_value=_goal_doc()), \
             patch.object(main.agent, "run", return_value=(_AGENT_GOAL_JSON, ["science.pdf"])):
            response = client.post("/goals/analyse", json={"user_id": "user-1"})
        assert "science.pdf" in response.json()["sources_used"]

    def test_missing_user_id_in_body_returns_422(self):
        response = client.post("/goals/analyse", json={})
        assert response.status_code == 422

    def test_empty_user_id_returns_422(self):
        response = client.post("/goals/analyse", json={"user_id": "   "})
        assert response.status_code == 422

    def test_no_goal_returns_404(self):
        with patch("main.get_goal", return_value=None):
            response = client.post("/goals/analyse", json={"user_id": "user-1"})
        assert response.status_code == 404

    def test_agent_non_json_response_returns_500(self):
        with patch("main.get_goal", return_value=_goal_doc()), \
             patch.object(main.agent, "run", return_value=("not json at all", [])):
            response = client.post("/goals/analyse", json={"user_id": "user-1"})
        assert response.status_code == 500
