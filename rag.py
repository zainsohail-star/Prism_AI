import os
from typing import Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

INDEX_PATH = "./faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 4

PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """You are a precise research assistant. Answer the question using ONLY the context below.
After your answer, state which section or page of the document you drew from.
If the context does not contain enough information, say "I don't have enough context to answer."

Context:
{context}

Question: {question}

Answer:"""
)


def _load_retriever() -> Any:
    """Load the FAISS index and return a retriever."""
    if not os.path.isdir(INDEX_PATH):
        raise FileNotFoundError("No index found. Run ingestion.py first.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return vector_store.as_retriever(search_kwargs={"k": TOP_K})


def _format_docs(docs: list) -> str:
    """Concatenate document page contents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def get_answer(question: str) -> dict:
    """
    Retrieve relevant chunks and generate a grounded answer.

    Args:
        question: The user's question string.

    Returns:
        A dict with:
            "answer"  (str)       – the LLM's response.
            "sources" (list[str]) – source metadata for each retrieved chunk.
    """
    retriever = _load_retriever()

    # --- Retrieve chunks for the question ---
    retrieved_docs = retriever.invoke(question)

    # --- Build sources list from metadata ---
    sources: list[str] = []
    for doc in retrieved_docs:
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "unknown")
        sources.append(f"{source} — page {page}")

    # --- RAG chain ---
    llm = ChatGroq(model=LLM_MODEL, temperature=0)

    chain = (
        {"context": lambda _: _format_docs(retrieved_docs), "question": RunnablePassthrough()}
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    answer: str = chain.invoke(question)

    return {"answer": answer, "sources": sources}
