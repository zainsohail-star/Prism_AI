import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

load_dotenv()


class AgentState(TypedDict):
    question: str
    search_results: list[dict]
    answer: str


def search_node(state: AgentState) -> AgentState:
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    raw = client.search(state["question"], max_results=3)
    results = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in raw.get("results", [])
    ]
    return {**state, "search_results": results}


def answer_node(state: AgentState) -> AgentState:
    context = "\n\n".join(
        f"[{r['title']}] ({r['url']})\n{r['content']}" for r in state["search_results"]
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response = llm.invoke(
        f"Answer using ONLY the search results below. "
        f"Cite each source URL inline after the claim it supports.\n\n"
        f"Search results: {context}\n\n"
        f"Question: {state['question']}"
    )
    return {**state, "answer": response.content}


# Build graph
_builder = StateGraph(AgentState)
_builder.add_node("search", search_node)
_builder.add_node("answer", answer_node)
_builder.set_entry_point("search")
_builder.add_edge("search", "answer")
_builder.add_edge("answer", END)
_graph = _builder.compile()


def web_search_answer(question: str) -> dict:
    result = _graph.invoke({"question": question, "search_results": [], "answer": ""})
    sources = [r["url"] for r in result["search_results"]]
    return {"answer": result["answer"], "sources": sources}
