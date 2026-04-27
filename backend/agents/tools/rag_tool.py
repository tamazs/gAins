from langchain_core.tools import tool

from tools.embedder import embed_text
from tools.mongo_vector_store import similarity_search


@tool
def rag_tool(query: str) -> str:
    """
    Search the training science knowledge base for information relevant to the query.
    Use this whenever you need evidence-based guidance on programming, periodisation,
    exercise selection, rep ranges, recovery, or nutrition.
    """
    query_embedding = embed_text(query)
    results = similarity_search(query_embedding, top_k=3)

    if not results:
        return "No relevant documents found."

    sections = []
    for text, score, source in results:
        sections.append(f"[Source: {source} | similarity: {score:.2f}]\n{text}")

    return "\n\n---\n\n".join(sections)
