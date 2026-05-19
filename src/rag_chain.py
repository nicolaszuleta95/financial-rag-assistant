"""
rag_chain.py — Core RAG pipeline for the Financial RAG Assistant.

FinancialRAGChain wraps LangChain's ConversationalRetrievalChain, wiring
together the ChromaDB vectorstore, gpt-4o-mini, and ConversationBufferWindowMemory.
It exposes a simple .ask() interface used by both the notebooks and the
Streamlit app.
"""

from __future__ import annotations

import tiktoken
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from src.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT

load_dotenv()

_LLM_MODEL = "gpt-4o-mini"
_TEMPERATURE = 0
_MAX_TOKENS = 500
_TOP_K = 4
_MEMORY_WINDOW = 5  # conversation turns to keep in memory


class FinancialRAGChain:
    """Full RAG pipeline for financial 10-K document analysis.

    Attributes:
        vectorstore: Chroma collection loaded from disk.
        llm: ChatOpenAI instance (gpt-4o-mini, temperature=0).
        memory: ConversationBufferMemory — tracks conversation history so
            follow-up questions work correctly.
        chain: ConversationalRetrievalChain — the assembled LangChain chain.
    """

    def __init__(
        self,
        vectorstore: Chroma,
        bank_filter: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialise the RAG chain.

        Args:
            vectorstore: Loaded Chroma vectorstore.
            bank_filter: When set, retrieval is restricted to documents whose
                ``bank`` metadata field equals this value (e.g. "JPMorgan Chase").
        """
        self.vectorstore = vectorstore
        self._bank_filter = bank_filter

        self._tokenizer = tiktoken.encoding_for_model(_LLM_MODEL)

        self.llm = ChatOpenAI(
            model=_LLM_MODEL,
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
            api_key=api_key,  # None → falls back to OPENAI_API_KEY env var (local/.env)
        )

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=_MEMORY_WINDOW,
        )

        # Build retriever — apply metadata filter when requested
        search_kwargs: dict = {"k": _TOP_K}
        if bank_filter:
            search_kwargs["filter"] = {"bank": bank_filter}

        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            condense_question_prompt=CONDENSE_QUESTION_PROMPT,
            return_source_documents=True,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ask(self, question: str) -> dict:
        """Ask a question and get an answer with source citations.

        Args:
            question: Natural-language question about the 10-K filings.

        Returns:
            Dict with keys:
                - ``answer`` (str): LLM-generated response.
                - ``sources`` (list[dict]): Each retrieved chunk as
                  ``{bank, year, page, text}``.
                - ``tokens_used`` (int): Approximate tokens consumed
                  (input + output) for cost tracking.
        """
        try:
            result = self.chain.invoke({"question": question})
        except Exception as exc:
            return {
                "answer": f"Error generating response: {exc}",
                "sources": [],
                "tokens_used": 0,
            }

        sources = [
            {
                "bank": doc.metadata.get("bank", "Unknown"),
                "year": doc.metadata.get("year", "N/A"),
                "page": doc.metadata.get("page", "N/A"),
                "text": doc.page_content,
            }
            for doc in result.get("source_documents", [])
        ]

        answer_text: str = result.get("answer", "")
        context_text = " ".join(doc.page_content for doc in result.get("source_documents", []))
        tokens_used = len(self._tokenizer.encode(question + context_text + answer_text))

        return {
            "answer": answer_text,
            "sources": sources,
            "tokens_used": tokens_used,
        }

    def clear_memory(self) -> None:
        """Clear the conversation history so a fresh session begins."""
        self.memory.clear()

    def get_source_documents(self, question: str) -> list:
        """Return retrieved chunks without generating an answer.

        Useful for debugging retrieval quality.

        Args:
            question: Query string.

        Returns:
            List of LangChain Document objects.
        """
        search_kwargs: dict = {"k": _TOP_K}
        if self._bank_filter:
            search_kwargs["filter"] = {"bank": self._bank_filter}

        return self.vectorstore.similarity_search(
            query=question,
            **search_kwargs,
        )
