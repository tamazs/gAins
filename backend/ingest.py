"""
ingest.py — RAG document ingestion pipeline

Reads every PDF in rag_docs/, splits them into chunks, embeds each chunk,
and stores it in MongoDB so the agent can retrieve them later.

Usage:
    python ingest.py              # index all PDFs in rag_docs/
    python ingest.py --clear      # wipe the collection first, then re-index
"""

import argparse
import os
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv

from tools.text_splitter import split_text
from tools.embedder import embed_texts
from tools.mongo_vector_store import store_document, clear_documents, count_documents

load_dotenv()

RAG_DOCS_DIR = Path(__file__).parent / "rag_docs"


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Pull plain text out of a PDF file."""
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def ingest_file(pdf_path: Path) -> int:
    """
    Full pipeline for one PDF:
      1. Extract text
      2. Split into overlapping chunks
      3. Embed all chunks in one batch
      4. Store each (chunk, embedding) pair in MongoDB

    Returns the number of chunks stored.
    """
    print(f"  Reading: {pdf_path.name}")
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        print(f"  WARNING: no text extracted from {pdf_path.name} — skipping")
        return 0

    chunks = split_text(text)
    print(f"  Split into {len(chunks)} chunks")

    print(f"  Embedding {len(chunks)} chunks (this may take a moment)...")
    embeddings = embed_texts(chunks)

    source = pdf_path.name
    for chunk, embedding in zip(chunks, embeddings):
        store_document(text=chunk, embedding=embedding, source=source)

    print(f"  Stored {len(chunks)} chunks from '{source}'")
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Ingest RAG documents into MongoDB")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete all existing documents before ingesting",
    )
    args = parser.parse_args()

    if args.clear:
        deleted = clear_documents()
        print(f"Cleared {deleted} existing documents from the collection.\n")

    pdfs = list(RAG_DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {RAG_DOCS_DIR}")
        return

    print(f"Found {len(pdfs)} PDF(s) in rag_docs/\n")

    total_chunks = 0
    for pdf_path in pdfs:
        total_chunks += ingest_file(pdf_path)
        print()

    print(f"Done. Total chunks in DB: {count_documents()} (added {total_chunks} this run)")


if __name__ == "__main__":
    main()
