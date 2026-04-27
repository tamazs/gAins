import os
from typing import Optional, List
from pymongo import MongoClient, DESCENDING
from pymongo.collection import Collection

_collection: Optional[Collection] = None


def _get_collection() -> Collection:
    global _collection
    if _collection is None:
        uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("DB_NAME")
        if not uri or not db_name:
            raise RuntimeError("MONGODB_URI and DB_NAME must be set before using the session store.")
        client = MongoClient(uri)
        _collection = client[db_name]["gym_sessions"]
    return _collection


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
