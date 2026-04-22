import os
from typing import List, Tuple
import numpy as np
from pymongo import MongoClient

# Configuration — override via environment variables
MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("DB_NAME")

_client = MongoClient(MONGO_URI)
_db = _client[DB_NAME]
_collection = _db["rag_documents"]  # separate collection from your sessions/goals


# --- Write operations ---

def store_document(text: str, embedding: List[float], source: str = "manual") -> str:
    """
    Persist a text chunk and its embedding vector in MongoDB.

    Each document in the collection has the shape:
        { text: str, embedding: [float], source: str }
    """
    doc = {
        "text": text,
        "embedding": embedding,
        "source": source,
    }
    result = _collection.insert_one(doc)
    return str(result.inserted_id)


def clear_documents() -> int:
    """Delete all stored documents. Returns the number of deleted documents."""
    result = _collection.delete_many({})
    return result.deleted_count


def count_documents() -> int:
    """Return the total number of indexed chunks."""
    return _collection.count_documents({})


# --- Read / search operations ---

def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Cosine similarity between two vectors.

    Returns a value in [-1, 1] where 1 = identical direction.
    This is the standard metric for dense embedding comparison.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def similarity_search(
    query_embedding: List[float], top_k: int = 3
) -> List[Tuple[str, float, str]]:
    """
    Find the top-k most similar chunks to a query embedding.

    Strategy: load all stored embeddings, compute cosine similarity in-memory,
    then return the highest-scoring chunks.

    Note: this works well for small-to-medium corpora. For large-scale use,
    replace this with MongoDB Atlas Vector Search ($vectorSearch aggregation).

    Returns:
        A list of (text, score, source) tuples, sorted by score descending.
    """
    docs = list(
        _collection.find({}, {"text": 1, "embedding": 1, "source": 1, "_id": 0})
    )

    if not docs:
        return []

    scored = [
        (
            doc["text"],
            _cosine_similarity(query_embedding, doc["embedding"]),
            doc.get("source", "unknown"),
        )
        for doc in docs
    ]

    # Sort by similarity score, highest first
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]