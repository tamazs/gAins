from typing import List


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split a long text into overlapping chunks.

    Overlap ensures that context at chunk boundaries is not lost — a sentence
    that spans two chunks will appear (partially) in both, so retrieval still
    finds it regardless of which chunk the query matches.

    Args:
        text:       The raw text to split.
        chunk_size: Maximum number of characters per chunk.
        overlap:    Number of characters repeated at the start of the next chunk.

    Returns:
        A list of text chunks.
    """
    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        # Move forward by (chunk_size - overlap) so the next chunk
        # re-includes the last `overlap` characters of the current one.
        start += chunk_size - overlap

    # Remove empty chunks that can appear at the very end
    return [c for c in chunks if c]