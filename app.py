
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from tavily import TavilyClient
from datetime import date
import os

st.set_page_config(page_title="News Agent", page_icon="📰", layout="centered")
st.title("Truthfeed AI")
st.caption("Real-time verified news + background knowledge")

GROQ_API_KEY   = st.secrets["GROQ_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    temperature=0,
    max_retries=2,
    max_tokens=400,
)

@tool
def search_web(query: str) -> str:
    """Search the web for latest news and information."""
    try:
        today = date.today().strftime("%B %d, %Y")
        dated_query = f"{query} {today}"
        response = tavily_client.search(
            query=dated_query,
            include_answer="advanced",
            search_depth="advanced",
            include_images=False,
            topic="news",
            days=1,
        )
        answer  = response.get("answer", "")
        results = response.get("results", [])
        filtered = []
        for r in results:
            published = r.get("published_date", "")
            content   = r.get("content", "")[:200]
            filtered.append(f"[{published}] {content}")
        full = f"{answer}\n\n" + "\n".join(filtered)
        return full[:800]
    except Exception as e:
        return f"Search failed: {str(e)}"

tools = [search_web]

system_prompt = SystemMessage(content="""You are an expert news analyst and knowledge assistant.

RULES:
- For today's news → always search first
- For background/context → use your own knowledge
- NEVER make up today's news without searching
- NEVER say "as of my knowledge cutoff"
- NEVER show [LIVE] or [KNOWLEDGE] labels

RESPONSE FORMAT:
Use only these subheadings with bullet points:

📰 Top Headlines
- bullet point facts

🔍 Key Details
- bullet point explanations

💡 Why It Matters
- bullet point only if relevant

RULES FOR BULLETS:
- Each bullet = one clear fact
- Keep each bullet short and simple
- Mention source only if important (e.g. "Reuters reports...")
- Max 5 bullets per section
- No long paragraphs

EXAMPLE:
User: "what is the news today"

📰 Top Headlines
- LIRR workers went on strike halting 300,000 commuters
- London police deployed 4,000 officers for two rallies
- NSW Liberal members defected to One Nation

🔍 Key Details
- Strike began midnight Saturday over wage disputes
- London protests include far-right and pro-Palestine marches

💡 Why It Matters
- LIRR strike could impact 250,000 Monday commuters
""")

llm_with_tools = llm.bind_tools(tools)
agent = create_react_agent(llm_with_tools, tools=tools, prompt=system_prompt)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask me anything about today's news..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Searching & analyzing..."):
            try:
                trimmed_history = st.session_state.chat_history[-4:]
                messages = [system_prompt] + trimmed_history + [("user", user_input)]
                response = agent.invoke({"messages": messages})
                reply = response["messages"][-1].content
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("assistant", reply))
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.markdown(reply)
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")

with st.sidebar:
    st.header("ℹ️ About")
    st.write("Real-time news via Tavily + LLaMA 3.1 on Groq")
    st.write(f"**Date:** {date.today().strftime('%B %d, %Y')}")
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
