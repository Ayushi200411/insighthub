import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.router import route
import tempfile

st.set_page_config(page_title="InsightHub", page_icon="🔎", layout="wide")

st.title("🔎 InsightHub")
st.caption("Multi-agent AI assistant — document Q&A (RAG), summarization, and data analysis")

# Sidebar: file upload for the Data agent
st.sidebar.header("Data Analysis")
uploaded_file = st.sidebar.file_uploader("Upload a CSV for analysis", type=["csv"])
target_column = None
csv_path = None

if uploaded_file is not None:
    temp_dir = tempfile.gettempdir()
    csv_path = os.path.join(temp_dir, uploaded_file.name)
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
    target_column = st.sidebar.text_input("Column to predict (optional, for ML baseline)")

st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info(
    "This system routes your question to one of three agents:\n\n"
    "- **Research agent** — answers from ingested documents (RAG)\n"
    "- **Summarizer agent** — condenses text you paste in\n"
    "- **Data agent** — runs EDA + ML on an uploaded CSV"
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "category" in msg:
            st.caption(f"Routed to: {msg['category']}")

# Chat input
user_input = st.chat_input("Ask a question, request a summary, or ask about your uploaded data...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Special handling: if a CSV is uploaded and question seems data-related,
            # pass the actual uploaded file path instead of the hardcoded default
            if csv_path and any(word in user_input.lower() for word in ["csv", "data", "dataset", "analyze", "predict"]):
                from agents.data_agent import run_data_agent
                result = run_data_agent(csv_path, target_column if target_column else None)
                category = "DATA (uploaded file)"
            else:
                result = route(user_input)
                category = "auto-routed"

        st.markdown(result["answer"])
        if "sources" in result:
            st.caption(f"Sources: {', '.join(result['sources'])}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "category": category
    })