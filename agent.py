# build_graph.py

import os

# Load .env locally (will be ignored safely on Streamlit Cloud)
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# ✅ New Tavily package
from langchain_tavily import TavilySearch

from langchain_groq import ChatGroq


def build_graph():

    # -------- READ API KEYS SAFELY --------
    try:
        tavily_key = os.environ["TAVILY_API_KEY"]
        groq_key = os.environ["GROQ_API_KEY"]
    except KeyError as e:
        raise RuntimeError(
            f"Missing required environment variable: {e}. "
            "Add it to Streamlit Secrets or your .env file."
        )

    # -------- TOOLS --------
    api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
    arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="query arxiv papers")

    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

    # ✅ MUST pass API key explicitly (new LangChain requirement)
    tavily = TavilySearch(api_key=tavily_key)

    tools = [arxiv, wiki, tavily]

    # -------- LLM --------
    llm = ChatGroq(
        model="qwen/qwen3-32b",
        api_key=groq_key  # ✅ Explicit injection required in cloud
    )

    llm_with_tools = llm.bind_tools(tools=tools)

    # -------- STATE --------
    class State(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

    # -------- NODE --------
    def tools_calling_llm(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # -------- GRAPH --------
    builder = StateGraph(State)

    builder.add_node("tools_calling_llm", tools_calling_llm)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "tools_calling_llm")
    builder.add_conditional_edges("tools_calling_llm", tools_condition)

    # Agent loop (reason → tool → reason)
    builder.add_edge("tools", "tools_calling_llm")
    builder.add_edge("tools_calling_llm", END)

    graph = builder.compile()

    return graph
