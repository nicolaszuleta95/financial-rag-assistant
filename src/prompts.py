"""
prompts.py — Prompt templates for the Financial RAG Assistant.

Defines the system prompt and the two LangChain PromptTemplates required by
ConversationalRetrievalChain:
  - QA_PROMPT: used to generate the final answer.
  - CONDENSE_QUESTION_PROMPT: condenses follow-up questions into standalone
    queries before retrieval.
"""

from langchain.prompts import PromptTemplate

# ---------------------------------------------------------------------------
# System / instruction text
# ---------------------------------------------------------------------------

FINANCIAL_QA_SYSTEM_PROMPT = """You are a financial analyst assistant specialized in banking annual reports (10-K filings).

Your task is to answer questions about the provided financial documents accurately and professionally.

STRICT RULES:
1. Answer ONLY based on the context provided below. Do not use external knowledge.
2. Always cite your sources: mention the bank name, year, and page number for each fact.
3. If the information is not in the provided context, say exactly:
   "I don't have that information in the available documents."
4. Use professional financial language.
5. Be concise but complete. Use bullet points for lists and format numbers properly.
6. If comparing multiple banks, structure the answer clearly by bank.

Context from 10-K filings:
{context}
"""

# ---------------------------------------------------------------------------
# QA prompt — generates the final answer from context + question
# ---------------------------------------------------------------------------

_QA_TEMPLATE = (
    FINANCIAL_QA_SYSTEM_PROMPT
    + "\n\nQuestion: {question}\n\nAnswer:"
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=_QA_TEMPLATE,
)

# ---------------------------------------------------------------------------
# Condense-question prompt — rewrites follow-up questions into standalone ones
# ---------------------------------------------------------------------------

_CONDENSE_TEMPLATE = """Given the following conversation history and a follow-up question, \
rephrase the follow-up question to be a self-contained standalone question that includes \
all necessary context from the conversation history.

If the follow-up is already standalone, return it unchanged.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=_CONDENSE_TEMPLATE,
)
