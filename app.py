import streamlit as st
import os

from src.config import GROQ_API_KEY, GOOGLE_API_KEY
from src.utils.helpers import compute_files_hash
from src.ui import render_chat_history, render_message_turn
from src.core.service import RagService
from src.ingestion.pipeline import ingest_documents

# Page Setup & Validation
st.set_page_config(
    page_title="Conversational PDF Self-RAG",
    layout="wide"
)

st.title("Conversational PDF Self-RAG Chatbot")
st.caption("Powered by LangGraph + Groq + ChromaDB")

if not GROQ_API_KEY:
    st.error("Missing Groq API Key. Please set GROQ_API_KEY in your .env file.")
    st.stop()

if not GOOGLE_API_KEY:
    st.warning("Missing Google API Key. Please set GOOGLE_API_KEY in your .env file for native Google Gemini Embeddings.")

# State Initialization
if "messages_by_session" not in st.session_state:
    st.session_state.messages_by_session = {}

if "uploaded_hash" not in st.session_state:
    st.session_state.uploaded_hash = None

if "is_ready" not in st.session_state:
    st.session_state.is_ready = False

if "rag_service" not in st.session_state:
    st.session_state.rag_service = RagService()

session_id = "default_session"

if session_id not in st.session_state.messages_by_session:
    st.session_state.messages_by_session[session_id] = []

session_messages = st.session_state.messages_by_session[session_id]

# PDF Upload & Ingestion Pipeline
uploaded_files = st.file_uploader(
    "Upload PDF Documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or multiple PDF documents to chat with."
)

if uploaded_files:
    current_hash = compute_files_hash(uploaded_files)
    if st.session_state.uploaded_hash != current_hash:
        with st.spinner("Uploading and indexing PDFs..."):
            try:
                retriever, chunk_count = ingest_documents(uploaded_files)
                st.session_state.rag_service.set_retriever(retriever)
                st.session_state.is_ready = True
                st.session_state.uploaded_hash = current_hash
                st.success(f"Processed {len(uploaded_files)} PDF(s) into {chunk_count} indexed chunks.")
            except Exception as e:
                st.error(f"Failed to process PDFs: {str(e)}")

st.divider()

# Conversation View & Query Execution
render_chat_history(session_messages)

user_query = st.chat_input("Ask a question about your PDF documents...")

if user_query:
    if not st.session_state.is_ready:
        st.warning("Please upload at least one PDF document before asking questions.")
    else:
        # Append and display user turn
        session_messages.append({"role": "user", "content": user_query})
        render_message_turn(role="user", content=user_query)

        # Execute query directly via RagService
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    result = st.session_state.rag_service.execute_query(
                        user_query=user_query,
                        session_id=session_id
                    )

                st.markdown(result.answer)



                # Persist assistant turn
                session_messages.append({
                    "role": "assistant",
                    "content": result.answer,
                    "reflections": result.reflection_logs,
                    "sources": result.sources
                })

            except Exception as e:
                st.error(f"Error during execution: {str(e)}")