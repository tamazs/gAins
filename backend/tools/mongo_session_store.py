import os
from typing import Optional, List
from pymongo import MongoClient, DESCENDING
from pymongo.collection import Collection

_sessions_collection: Optional[Collection] = None
_goals_collection: Optional[Collection] = None
_goal_entries_collection: Optional[Collection] = None


def _get_db():
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("DB_NAME")
    if not uri or not db_name:
        raise RuntimeError("MONGODB_URI and DB_NAME must be set before using the session store.")
    return MongoClient(uri)[db_name]


def _get_collection() -> Collection:
    global _sessions_collection
    if _sessions_collection is None:
        _sessions_collection = _get_db()["gym_sessions"]
    return _sessions_collection


def _get_goals_collection() -> Collection:
    global _goals_collection
    if _goals_collection is None:
        _goals_collection = _get_db()["user_goals"]
    return _goals_collection


def _get_goal_entries_collection() -> Collection:
    global _goal_entries_collection
    if _goal_entries_collection is None:
        _goal_entries_collection = _get_db()["goal_entries"]
    return _goal_entries_collection


def save_session(session_id: str, session: dict) -> None:
    """Persist a workout session to MongoDB."""
    _get_collection().insert_one({"session_id": session_id, **session})


def get_recent_sessions(user_id: str, limit: int = 5) -> List[dict]:
    """Fetch the most recent sessions for a user, newest first."""
    cursor = (
        _get_collection()
        .find({"user_id": user_id}, {"_id": 0, "embedding": 0})
        .sort("date", DESCENDING)
        .limit(limit)
    )
    return list(cursor)


# --- Goal storage ---

def save_goal(goal_id: str, goal: dict) -> None:
    """
    Upsert a training goal for a user.
    Each user has at most one active goal — saving a new one replaces the old one.
    """
    _get_goals_collection().replace_one(
        {"user_id": goal["user_id"]},
        {"goal_id": goal_id, **goal},
        upsert=True,
    )


def get_goal(user_id: str) -> Optional[dict]:
    """Retrieve the active goal for a user, or None if no goal is set."""
    return _get_goals_collection().find_one(
        {"user_id": user_id},
        {"_id": 0},
    )


# --- Goal entry storage ---

def save_goal_entry(entry_id: str, user_id: str, entry: dict) -> None:
    """Persist a single training entry logged toward the user's active goal."""
    _get_goal_entries_collection().insert_one(
        {"entry_id": entry_id, "user_id": user_id, **entry}
    )


def get_goal_entries(user_id: str, exercise_name: str, limit: int = 10) -> List[dict]:
    """Fetch the most recent entries for a user's goal exercise, newest first."""
    cursor = (
        _get_goal_entries_collection()
        .find(
            {"user_id": user_id, "exercise_name": exercise_name},
            {"_id": 0},
        )
        .sort("date", DESCENDING)
        .limit(limit)
    )
    return list(cursor)
