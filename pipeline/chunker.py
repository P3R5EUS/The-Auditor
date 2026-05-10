"""
chunker.py — Split long 10-K text into overlapping chunks for RAG.

10-K filings are 100-300 pages. We use RecursiveCharacterTextSplitter
which tries paragraph → sentence → word boundaries in order, so chunks
are semantically coherent.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 1500       # ~375 tokens — fits well in embedding models
CHUNK_OVERLAP = 200     # Overlap so context isn't lost at boundaries


def chunk_text(text: str) -> list[str]:
    """
    Split raw filing text into overlapping chunks.

    Returns:
        List of string chunks, ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    # Pre-clean: remove excessive whitespace / form-feed chars common in SEC filings
    text = _clean_text(text)

    chunks = splitter.split_text(text)
    print(f"[chunker] Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def _clean_text(text: str) -> str:
    """Remove noise common in SEC EDGAR plain-text filings."""
    import re
    # Remove form-feed characters
    text = text.replace("\x0c", "\n")
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove long dashes/underscores used as visual dividers
    text = re.sub(r"[-_=]{10,}", "", text)
    # Remove XBRL tags if any leaked through
    text = re.sub(r"<[^>]+>", " ", text)
    return text.strip()