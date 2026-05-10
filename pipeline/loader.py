"""
loader.py — Load 10-K filings from .txt or .pdf files.
Supports both plain text and PDF (via pdfminer).
"""

from pathlib import Path


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
    import pdfplumber
    text = ""
    with pdfplumber.open(str(filepath)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    print(f"[loader] Loaded PDF: {filepath.name} ({len(text):,} chars)")
    return text