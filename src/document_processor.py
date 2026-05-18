"""
document_processor.py — PDF extraction, cleaning, and chunking.

Loads 10-K filings page by page, cleans extracted text, splits into
overlapping chunks using LangChain's RecursiveCharacterTextSplitter,
and serializes results to JSON.
"""

import json
import re
import unicodedata
from pathlib import Path

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean text extracted from a PDF page.

    Removes non-printable characters, standalone page numbers, excessive
    whitespace, and hyphenated line breaks.

    Args:
        text: Raw string from pdfplumber.

    Returns:
        Cleaned string ready for chunking.
    """
    # Normalise unicode (e.g. ligatures, en-dashes)
    text = unicodedata.normalize("NFKC", text)

    # Remove non-printable / control characters (keep newlines and tabs)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", " ", text)

    # Rejoin hyphenated line breaks (e.g. "finan-\ncial" → "financial")
    text = re.sub(r"-\n(\S)", r"\1", text)

    # Remove lines that are just a number (standalone page numbers)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Collapse multiple blank lines into two at most
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces/tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Strip leading/trailing whitespace on each line
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(lines)

    return text.strip()


def is_valid_page(text: str, min_chars: int = 100) -> bool:
    """Return True if a page contains meaningful content.

    Args:
        text: Cleaned page text.
        min_chars: Minimum character count threshold.

    Returns:
        True when the page has at least 20 words and *min_chars* characters.
    """
    words = text.split()
    return len(words) >= 20 and len(text) >= min_chars


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(
    file_path: Path,
    bank: str,
    ticker: str,
    year: int,
) -> list[dict]:
    """Extract text from every page of a PDF with associated metadata.

    Args:
        file_path: Absolute path to the PDF file.
        bank: Full bank name (e.g. "JPMorgan Chase").
        ticker: Stock ticker symbol (e.g. "JPM").
        year: Filing year (e.g. 2023).

    Returns:
        List of dicts, one per valid page, with keys:
            bank, ticker, year, page, total_pages, text, word_count.
    """
    pages_data: list[dict] = []

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        for page_obj in pdf.pages:
            raw_text = page_obj.extract_text() or ""
            cleaned = clean_text(raw_text)

            if not is_valid_page(cleaned):
                continue

            pages_data.append(
                {
                    "bank": bank,
                    "ticker": ticker,
                    "year": year,
                    "page": page_obj.page_number,  # 1-indexed
                    "total_pages": total_pages,
                    "text": cleaned,
                    "word_count": len(cleaned.split()),
                }
            )

    return pages_data


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    """Count words in *text* (used as length_function for the splitter)."""
    return len(text.split())


def chunk_documents(
    pages_data: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Split page texts into overlapping chunks.

    Uses RecursiveCharacterTextSplitter configured to measure length in
    words, with a preference for splitting at paragraph/sentence boundaries.

    Args:
        pages_data: Output of :func:`extract_text_from_pdf`.
        chunk_size: Maximum words per chunk.
        chunk_overlap: Word overlap between adjacent chunks.

    Returns:
        List of chunk dicts with keys:
            chunk_id, bank, ticker, year, page, text, word_count, char_count.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_word_count,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks: list[dict] = []

    for page in pages_data:
        page_chunks = splitter.split_text(page["text"])

        for chunk_num, chunk_text in enumerate(page_chunks):
            word_count = len(chunk_text.split())
            if word_count < 30:
                continue  # skip fragments

            chunk_id = (
                f"{page['ticker']}_{page['year']}"
                f"_p{page['page']:04d}_c{chunk_num:06d}"
            )
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "bank": page["bank"],
                    "ticker": page["ticker"],
                    "year": page["year"],
                    "page": page["page"],
                    "text": chunk_text,
                    "word_count": word_count,
                    "char_count": len(chunk_text),
                }
            )

    return chunks


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_chunks(chunks: list[dict], output_path: Path) -> None:
    """Serialise chunks list to a JSON file.

    Args:
        chunks: List of chunk dicts.
        output_path: Destination file path (created if absent).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks)} chunks → {output_path}")


def load_chunks(path: Path) -> list[dict]:
    """Load chunks from a JSON file produced by :func:`save_chunks`.

    Args:
        path: Path to the JSON file.

    Returns:
        List of chunk dicts.
    """
    with open(path, "r", encoding="utf-8") as fh:
        chunks = json.load(fh)
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks
