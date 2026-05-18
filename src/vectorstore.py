"""
vectorstore.py — ChromaDB setup, ingestion, and semantic search.

Manages the local, persisted ChromaDB collection that stores chunk
embeddings. The collection is created once (notebook 02) and reloaded
from disk by the app on every startup — no re-embedding required.
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

_COLLECTION_NAME = "financial_rag"
_EMBEDDING_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def get_embeddings() -> OpenAIEmbeddings:
    """Return an OpenAIEmbeddings instance using *text-embedding-3-small*.

    Reads OPENAI_API_KEY from the environment (loaded via python-dotenv).

    Returns:
        Configured OpenAIEmbeddings object.
    """
    return OpenAIEmbeddings(model=_EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Create / load
# ---------------------------------------------------------------------------

def create_vectorstore(
    chunks: list[dict],
    persist_dir: str = "vectorstore",
    batch_size: int = 100,
) -> Chroma:
    """Create a ChromaDB collection from a list of chunk dicts.

    Each chunk's ``text`` field becomes the document content; the remaining
    fields become ChromaDB metadata. The collection is persisted to disk at
    *persist_dir* and returned ready for querying.

    Args:
        chunks: Output of document_processor.chunk_documents().
        persist_dir: Directory where ChromaDB stores its data files.
        batch_size: Number of chunks per embedding API call (avoids the
            300k-token-per-request limit).

    Returns:
        Chroma vectorstore instance.
    """
    embeddings = get_embeddings()

    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "bank": c["bank"],
            "ticker": c["ticker"],
            "year": c["year"],
            "page": c["page"],
            "word_count": c["word_count"],
            "chunk_id": c["chunk_id"],
        }
        for c in chunks
    ]
    ids = [c["chunk_id"] for c in chunks]

    # Create collection with the first batch, then add remaining batches.
    vectorstore = Chroma.from_texts(
        texts=texts[:batch_size],
        embedding=embeddings,
        metadatas=metadatas[:batch_size],
        ids=ids[:batch_size],
        collection_name=_COLLECTION_NAME,
        persist_directory=str(Path(persist_dir)),
    )

    for start in range(batch_size, len(texts), batch_size):
        end = start + batch_size
        vectorstore.add_texts(
            texts=texts[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        print(f"  Ingested {min(end, len(texts))}/{len(texts)} chunks…")

    print(
        f"Created vectorstore with {vectorstore._collection.count()} documents"
        f" → {persist_dir}/"
    )
    return vectorstore


def load_vectorstore(persist_dir: str = "vectorstore") -> Chroma:
    """Load an existing ChromaDB collection from disk.

    Use this in the Streamlit app — avoids re-embedding on every startup.

    Args:
        persist_dir: Directory used when the collection was created.

    Returns:
        Chroma vectorstore instance ready for querying.
    """
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(Path(persist_dir)),
    )

    count = vectorstore._collection.count()
    print(f"Loaded vectorstore — {count} documents from {persist_dir}/")
    return vectorstore


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    vectorstore: Chroma,
    query: str,
    k: int = 4,
    bank_filter: str | None = None,
) -> list:
    """Semantic similarity search with optional bank filter.

    Args:
        vectorstore: Loaded Chroma collection.
        query: Natural-language query string.
        k: Number of top results to return.
        bank_filter: If provided, restrict search to documents where
            ``metadata["bank"] == bank_filter``.

    Returns:
        List of LangChain Document objects with ``page_content`` and
        ``metadata`` attributes.
    """
    where = {"bank": bank_filter} if bank_filter else None

    return vectorstore.similarity_search(
        query=query,
        k=k,
        filter=where,
    )
