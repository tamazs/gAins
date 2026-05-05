import os
from typing import Optional
from pymongo import MongoClient
from pymongo.collection import Collection

_users_collection: Optional[Collection] = None


def _get_collection() -> Collection:
    global _users_collection
    if _users_collection is None:
        uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("DB_NAME")
        if not uri or not db_name:
            raise RuntimeError("MONGODB_URI and DB_NAME must be set.")
        _users_collection = MongoClient(uri)[db_name]["users"]
        # Enforce unique emails at the database level
        _users_collection.create_index("email", unique=True)
    return _users_collection


def create_user(user_id: str, email: str, username: str, hashed_password: str) -> None:
    """Insert a new user. Raises DuplicateKeyError if the email is already taken."""
    _get_collection().insert_one({
        "user_id": user_id,
        "email": email,
        "username": username,
        "hashed_password": hashed_password,
    })


def get_user_by_email(email: str) -> Optional[dict]:
    """Look up a user by email, or return None if not found."""
    return _get_collection().find_one({"email": email}, {"_id": 0})
