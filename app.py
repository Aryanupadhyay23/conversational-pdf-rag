import streamlit as st

from src.config import GROQ_API_KEY, GOOGLE_API_KEY
from src.utils.helpers import compute_files_hash
from src.ingestion import ingest_documents
from src.core import RagService
from src.ui import render_sidebar, render_chat_history, render_message_turn

# ---------------------------------------------------------
# Page Setup & Validation
# ---------------------------------------------------------
st.set_page_config(
    page_title="Conversational PDF Self-RAG",
    layout="wide"
)

st.title("Conversational PDF Self-RAG Chatbot")
st.caption("Modular asynchronous architecture with LangGraph Memory Checkpointer, Groq LLM, and Self-RAG")

if not GROQ_API_KEY:
    st.error("Missing Groq API Key. Please set GROQ_API_KEY in your .env file.")
    st.stop()

if not GOOGLE_API_KEY:
    st.warning("Missing Google API Key. Please set GOOGLE_API_KEY in your .env file for native Google Gemini Embeddings.")

# ---------------------------------------------------------
# State & Service Initialization
# ---------------------------------------------------------
if "rag_service" not in st.session_state:
    st.session_state.rag_service = RagService()

if "messages_by_session" not in st.session_state:
    st.session_state.messages_by_session = {}

if "uploaded_hash" not in st.session_state:
    st.session_state.uploaded_hash = None

rag_service: RagService = st.session_state.rag_service

# Render sidebar controls & get active session ID
session_id = render_sidebar(rag_service)

if session_id not in st.session_state.messages_by_session:
    st.session_state.messages_by_session[session_id] = []

session_messages = st.session_state.messages_by_session[session_id]

# ---------------------------------------------------------
# PDF Upload & Ingestion Pipeline
# ---------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or multiple PDF documents to chat with."
)

if uploaded_files:
    current_hash = compute_files_hash(uploaded_files)
    if st.session_state.uploaded_hash != current_hash:
        with st.spinner("Ingesting and indexing PDF documents..."):
            retriever, chunk_count = ingest_documents(uploaded_files)
            rag_service.set_retriever(retriever)
            st.session_state.uploaded_hash = current_hash
            st.success(f"Processed {len(uploaded_files)} PDF(s) into {chunk_count} indexed chunks.")

st.divider()

# ---------------------------------------------------------
# Conversation View & Query Execution
# ---------------------------------------------------------
render_chat_history(session_messages)

user_query = st.chat_input("Ask a question about your PDF documents...")

if user_query:
    if not rag_service.is_ready():
        st.warning("Please upload at least one PDF document before asking questions.")
    else:
        # Append and display user turn
        session_messages.append({"role": "user", "content": user_query})
        render_message_turn(role="user", content=user_query)

        # Execute query via RagService
        with st.chat_message("assistant"):
            status_container = st.status("Self-RAG Agent Reflecting...", expanded=True)
            status_container.write("Invoking Async Self-RAG Graph with Conversation Memory...")

            try:
                result = rag_service.execute_query(user_query, session_id=session_id)

                for log_entry in result.reflection_logs:
                    status_container.markdown(log_entry)
                status_container.update(label="Self-RAG Reflection Complete", state="complete", expanded=False)

                st.markdown(result.answer)

                if result.sources:
                    with st.expander("Retrieved Source Passages", expanded=False):
                        for s in result.sources:
                            st.markdown(f"**{s['source']} (Page {s['page']})**")
                            st.markdown(f"> {s['snippet']}")

                # Persist assistant turn in session
                session_messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "reflections": result.reflection_logs,
                    "sources": result.sources
                })

            except Exception as e:
                status_container.update(label="Self-RAG Error", state="error", expanded=True)
                st.error(f"Error during Self-RAG execution: {str(e)}")