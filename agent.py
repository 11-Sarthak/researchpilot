# build_graph.py

from dotenv import load_dotenv
load_dotenv()

from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper

# ⚠️ Update Tavily import (fix deprecation)
from langchain_tavily import TavilySearch

from langchain_groq import ChatGroq


def build_graph():

    # -------- TOOLS --------
    api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500)
    arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv, description="query arxiv papers")

    api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

    tavily = TavilySearch()

    tools = [arxiv, wiki, tavily]

    # -------- LLM --------
    llm = ChatGroq(model="qwen/qwen3-32b")
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

    # loop so agent can think after tool call
    builder.add_edge("tools", "tools_calling_llm")
    builder.add_edge("tools_calling_llm", END)

    graph = builder.compile()

    return graph
