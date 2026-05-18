"""
conftest.py — Shared pytest fixtures and import-time stubs.

CRITICAL: The sys.modules stubs at the top of this file MUST execute before any
local module is imported.  They prevent ChatOllama / OllamaEmbeddings from
attempting to connect to a running Ollama server during test collection.

langchain_core is intentionally left real so that:
  • BaseCallbackHandler is a proper base class (needed by _SourceTracker)
  • @tool decorators work normally in agent tool tests
"""

import sys
from unittest.mock import MagicMock

# ── Pre-import stubs ──────────────────────────────────────────────────────────
# Only stub the packages that contain network-connecting constructors.
for _mod in [
    "langchain_ollama",
    "langchain_classic",
    "langchain_classic.agents",
    "mcp",
    "mcp.client",
    "mcp.client.stdio",
    "langchain_mcp_adapters",
    "langchain_mcp_adapters.tools",
]:
    sys.modules.setdefault(_mod, MagicMock())

# ── Standard library / pytest imports (AFTER stubs) ──────────────────────────
import pytest


# ── JWT env-var fixture ───────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def set_jwt_secret(monkeypatch):
    """Ensure JWT_SECRET is set for every test that touches token helpers."""
    monkeypatch.setenv("JWT_SECRET", "test-secret-key-for-unit-tests")


# ── MongoDB singleton reset ───────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_mongo_singletons():
    """
    Reset all module-level MongoDB collection singletons to None before and
    after every test so that lazy-init re-runs inside each test's mock context.
    Without this, the first real (or mocked) MongoClient call would pollute
    subsequent tests.
    """
    import tools.mongo_session_store as mss
    import tools.mongo_user_store as mus
    import tools.mongo_vector_store as mvs

    def _reset():
        mss._sessions_collection = None
        mss._goals_collection = None
        mss._goal_entries_collection = None
        mus._users_collection = None
        mvs._collection = None

    _reset()
    yield
    _reset()


# ── Reusable payload fixtures ─────────────────────────────────────────────────

_PAST_DATE = "2024-06-15T10:00:00Z"
_FUTURE_DEADLINE = "2030-12-31"


@pytest.fixture
def sample_set():
    return {"reps": 5, "weight_kg": 100.0, "rpe": 8.0}


@pytest.fixture
def sample_exercise():
    return {
        "name": "Bench Press",
        "muscle_group": "chest",
        "sets": [{"reps": 5, "weight_kg": 100.0}],
    }


@pytest.fixture
def sample_session_payload():
    return {
        "user_id": "user-123",
        "date": _PAST_DATE,
        "exercises": [
            {
                "name": "Bench Press",
                "muscle_group": "chest",
                "sets": [{"reps": 5, "weight_kg": 100.0}],
            }
        ],
        "notes": "Felt good",
    }


@pytest.fixture
def sample_goal_payload():
    return {
        "user_id": "user-123",
        "exercise_name": "Squat",
        "muscle_group": "quads",
        "target_weight_kg": 140.0,
        "target_reps": 1,
        "deadline": _FUTURE_DEADLINE,
        "notes": "Preparing for meet",
    }


@pytest.fixture
def sample_goal_entry_payload():
    return {
        "user_id": "user-123",
        "date": _PAST_DATE,
        "sets": [{"reps": 3, "weight_kg": 130.0}],
        "notes": "Heavy single",
    }


@pytest.fixture
def sample_goal_doc():
    return {
        "goal_id": "goal-abc",
        "user_id": "user-123",
        "exercise_name": "Squat",
        "muscle_group": "quads",
        "target_weight_kg": 140.0,
        "target_reps": 1,
        "deadline": None,
        "notes": None,
        "created_at": "2025-01-01T00:00:00",
    }
