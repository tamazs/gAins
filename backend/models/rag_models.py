from pydantic import BaseModel
from typing import List, Optional


# --- Step 1: Indexing models ---

class IndexRequest(BaseModel):
    text: str
    source: Optional[str] = "manual"


class IndexResponse(BaseModel):
    message: str
    chunks_indexed: int


# --- Step 2: Retrieval models ---

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3


class RetrievedChunk(BaseModel):
    text: str
    score: float
    source: Optional[str] = "unknown"


class RetrieveResponse(BaseModel):
    query: str
    chunks: List[RetrievedChunk]


# --- Step 3 + 4: Augmented Generation models ---

class AskRequest(BaseModel):
    question: str
    top_k: int = 3


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    augmented_prompt: str


# --- Utility models ---

class DocumentCountResponse(BaseModel):
    count: int


class ClearResponse(BaseModel):
    message: str
    deleted: int