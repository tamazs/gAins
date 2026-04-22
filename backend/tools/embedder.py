from langchain_ollama import OllamaEmbeddings
from typing import List

# nomic-embed-text is a fast, high-quality embedding model available in Ollama.
# Run: ollama pull nomic-embed-text
EMBEDDING_MODEL = "nomic-embed-text"

_embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)


def embed_text(text: str) -> List[float]:
    """Embed a single string (e.g. a query or a chunk)."""
    return _embedder.embed_query(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings in one batch (more efficient for indexing)."""
    return _embedder.embed_documents(texts)