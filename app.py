
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent import build_graph

load_dotenv()

st.set_page_config(page_title="ResearchPilot", page_icon="🔎", layout="wide")

# Build graph once (cached)
@st.cache_resource
def load_agent():
    return build_graph()

graph = load_agent()

st.title("🔎 ResearchPilot")
st.caption("AI Research Assistant (Arxiv • Wikipedia • Web)")

if "chat" not in st.session_state:
    st.session_state.chat = []

# Display chat
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)

# Input
query = st.chat_input("Ask a research question...")

if query:
    st.session_state.chat.append(("user", query))

    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            result = graph.invoke({"messages": [HumanMessage(content=query)]})
            answer = result["messages"][-1].content
            st.markdown(answer)

    st.session_state.chat.append(("assistant", answer))
