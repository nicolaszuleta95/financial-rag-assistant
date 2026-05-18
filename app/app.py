"""
app.py — Streamlit chatbot for the Financial RAG Assistant.

Sections:
  - Sidebar: bank selector, token counter, clear button, suggested questions.
  - Main: conversational chat interface with sources panel.
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Allow imports from the project root when running via `streamlit run app/app.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="🏦",
    layout="wide",
)

# Streamlit Cloud: read API key from secrets; local: read from .env
_secrets_paths = [
    Path.home() / ".streamlit" / "secrets.toml",
    ROOT / ".streamlit" / "secrets.toml",
]
if any(p.exists() for p in _secrets_paths):
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from src.rag_chain import FinancialRAGChain  # noqa: E402
from src.vectorstore import load_vectorstore  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANK_OPTIONS = {
    "All Banks": None,
    "JPMorgan Chase": "JPMorgan Chase",
    "Bank of America": "Bank of America",
    "Wells Fargo": "Wells Fargo",
}

SUGGESTED_QUESTIONS = [
    "What was JPMorgan's net revenue in 2023?",
    "What were JPMorgan's top 3 risk factors?",
    "Compare revenue growth between JPMorgan and Bank of America",
    "What is Wells Fargo's digital transformation strategy?",
    "How did interest rates affect net interest income?",
]

VECTORSTORE_DIR = str(ROOT / "vectorstore")

# ---------------------------------------------------------------------------
# Cached resources — loaded once per session
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading vectorstore…")
def _load_vectorstore():
    return load_vectorstore(persist_dir=VECTORSTORE_DIR)


@st.cache_resource(show_spinner="Initialising RAG chain…")
def _load_chain(bank_filter: str | None):
    vs = _load_vectorstore()
    return FinancialRAGChain(vs, bank_filter=bank_filter)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "selected_bank" not in st.session_state:
    st.session_state.selected_bank = "All Banks"
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏦 Financial RAG")
    st.caption("10-K filings · 2023 · JPMorgan · BofA · Wells Fargo")
    st.divider()

    selected_bank_label = st.selectbox(
        "Filter by bank",
        options=list(BANK_OPTIONS.keys()),
        index=list(BANK_OPTIONS.keys()).index(st.session_state.selected_bank),
    )

    # Rebuild chain when bank selection changes
    if selected_bank_label != st.session_state.selected_bank:
        st.session_state.selected_bank = selected_bank_label
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        st.rerun()

    st.caption("**Year:** 2023")
    st.divider()

    if st.button("🗑 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_tokens = 0
        bank_filter = BANK_OPTIONS[st.session_state.selected_bank]
        chain = _load_chain(bank_filter)
        chain.clear_memory()
        st.rerun()

    st.metric("Tokens used (session)", st.session_state.total_tokens)
    st.divider()

    st.subheader("Suggested Questions")
    for q in SUGGESTED_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

    st.divider()
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)]"
        "(https://github.com/nicolaszuleta95/financial-rag-assistant)"
    )

# ---------------------------------------------------------------------------
# Main — chat interface
# ---------------------------------------------------------------------------

bank_filter = BANK_OPTIONS[st.session_state.selected_bank]
chain = _load_chain(bank_filter)

st.title("🏦 Financial RAG Assistant")
st.caption(
    "Ask any question about the JPMorgan Chase, Bank of America, and Wells Fargo "
    "2023 10-K filings. Answers are grounded in the documents — with source citations."
)

# Render conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📄 View Sources"):
                for src in message["sources"]:
                    st.markdown(
                        f"**{src['bank']} {src['year']} · Page {src['page']}**"
                    )
                    st.caption(src["text"][:300] + ("…" if len(src["text"]) > 300 else ""))
                    st.divider()

# Handle a question (either from chat input or suggested-question button)
user_input = st.chat_input("Ask a question about the 10-K filings…")

# Suggested-question button injects the question on next rerun
if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None

if user_input:
    if not user_input.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents…"):
            result = chain.ask(user_input)

        st.markdown(result["answer"])

        if result["sources"]:
            with st.expander("📄 View Sources"):
                for src in result["sources"]:
                    st.markdown(
                        f"**{src['bank']} {src['year']} · Page {src['page']}**"
                    )
                    st.caption(
                        src["text"][:300] + ("…" if len(src["text"]) > 300 else "")
                    )
                    st.divider()

    # Update session state
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        }
    )
    st.session_state.total_tokens += result["tokens_used"]
    st.rerun()
