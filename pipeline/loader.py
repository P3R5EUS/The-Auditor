"""
loader.py — Load 10-K filings from .txt or .pdf files.
Supports both plain text and PDF (via pdfminer).
"""
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_filing(filepath: Path) -> str:
    """Load a filing and return its raw text content."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Filing not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == ".txt":
        return _load_txt(filepath)
    elif suffix == ".pdf":
        return _load_pdf(filepath)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .txt or .pdf")


def _load_txt(filepath: Path) -> str:
    """Load plain text filing."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    print(f"[loader] Loaded TXT: {filepath.name} ({len(text):,} chars)")
    return text


def _load_pdf(filepath: Path) -> str:
    print(f"[loader] Initializing PyPDFLoader for {filepath.name}...")
    loader = PyPDFLoader(str(filepath))
    pages = loader.load()
    text = "\n".join([page.page_content for page in pages])
    print(f"[loader] Successfully loaded {len(pages)} pages.")
    return text
