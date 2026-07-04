"""
Full Spectrum (Mode 4)
──────────────────────
The signature capability of PRISM AI: instead of picking ONE lens, this graph
runs all three — document, general knowledge, and live web — then passes all
three independent answers into a synthesis node that cross-checks them and
returns one confident, unified answer. Where the lenses agree, the answer is
reinforced. Where they disagree, the synthesis says so instead of hiding it.

This mirrors the physical metaphor the product is named for: a prism splits
one beam into a spectrum, and a converging lens can recombine that spectrum
back into a single, brighter point.
"""

from typing import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from agent import web_search_answer
from rag import get_answer

LLM_MODEL = "llama-3.3-70b-versatile"


class SpectrumState(TypedDict):
    question: str
    doc_answer: str
    doc_sources: list
    doc_available: bool
    knowledge_answer: str
    web_answer: str
    web_sources: list
    final_answer: str


def _doc_node(state: SpectrumState) -> SpectrumState:
    """Lens 1 — try the uploaded document. Missing/empty index is not an error here."""
    try:
        result = get_answer(state["question"])
        return {**state, "doc_answer": result["answer"], "doc_sources": result["sources"], "doc_available": True}
    except FileNotFoundError:
        return {**state, "doc_answer": "", "doc_sources": [], "doc_available": False}


def _knowledge_node(state: SpectrumState) -> SpectrumState:
    """Lens 2 — the model's own general knowledge, independent of the document."""
    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    response = llm.invoke(
        "Answer the following question using your own general knowledge, in 3-4 sentences. "
        "Do not mention that you lack document context.\n\n"
        f"Question: {state['question']}"
    )
    return {**state, "knowledge_answer": response.content}


def _web_node(state: SpectrumState) -> SpectrumState:
    """Lens 3 — live web search via Tavily."""
    try:
        result = web_search_answer(state["question"])
        return {**state, "web_answer": result["answer"], "web_sources": result["sources"]}
    except Exception:
        return {**state, "web_answer": "", "web_sources": []}


def _synthesis_node(state: SpectrumState) -> SpectrumState:
    """Recombine all three lenses into one cross-verified answer."""
    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    doc_block = state["doc_answer"] if state["doc_available"] else "(no document uploaded)"
    prompt = (
        "You are given the SAME question answered independently from three different lenses: "
        "a private document, general knowledge, and a live web search. "
        "Write ONE final answer that merges them. "
        "If the lenses agree, state the answer with full confidence. "
        "If they conflict on a specific fact, briefly flag the disagreement instead of silently "
        "picking one side. Do not refer to 'Lens 1/2/3' by number — write it as a normal, "
        "well-organized answer a person would read.\n\n"
        f"Question: {state['question']}\n\n"
        f"Document lens: {doc_block}\n\n"
        f"General knowledge lens: {state['knowledge_answer']}\n\n"
        f"Web search lens: {state['web_answer'] or '(no web results)'}\n\n"
        "Final synthesized answer:"
    )
    response = llm.invoke(prompt)
    return {**state, "final_answer": response.content}


_builder = StateGraph(SpectrumState)
_builder.add_node("doc", _doc_node)
_builder.add_node("knowledge", _knowledge_node)
_builder.add_node("web", _web_node)
_builder.add_node("synthesize", _synthesis_node)

# Fan-out: doc / knowledge / web don't depend on each other, so they run
# concurrently in the same superstep instead of one-after-another. This is
# the main speed win for Mode 4 — three independent Groq/Tavily calls run
# in parallel instead of stacking their latencies end to end.
_builder.add_edge(START, "doc")
_builder.add_edge(START, "knowledge")
_builder.add_edge(START, "web")

# Fan-in: synthesize only runs once all three lenses have reported back.
_builder.add_edge("doc", "synthesize")
_builder.add_edge("knowledge", "synthesize")
_builder.add_edge("web", "synthesize")
_builder.add_edge("synthesize", END)
_graph = _builder.compile()


def full_spectrum_answer(question: str) -> dict:
    """
    Mode 4 — 'Full Spectrum'.

    Runs all three lenses and returns a synthesized answer, plus a
    breakdown of what each individual lens said (so the UI can show
    the working, not just the final answer).
    """
    result = _graph.invoke(
        {
            "question": question,
            "doc_answer": "",
            "doc_sources": [],
            "doc_available": False,
            "knowledge_answer": "",
            "web_answer": "",
            "web_sources": [],
            "final_answer": "",
        }
    )

    sources = list(result["doc_sources"]) + list(result["web_sources"])

    return {
        "answer": result["final_answer"],
        "sources": sources,
        "lenses": {
            "document": result["doc_answer"] if result["doc_available"] else None,
            "knowledge": result["knowledge_answer"],
            "web": result["web_answer"] or None,
        },
    }
