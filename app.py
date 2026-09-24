import streamlit as st
import requests
import json
import uuid

from src.config import GROQ_API_KEY, GOOGLE_API_KEY
from src.utils.helpers import compute_files_hash
from src.ui import render_chat_history, render_message_turn

# FastAPI Backend Configuration
API_URL = "http://127.0.0.1:8000"

# ---------------------------------------------------------
# Page Setup & Validation
# ---------------------------------------------------------
st.set_page_config(
    page_title="Conversational PDF Self-RAG",
    layout="wide"
)

st.title("Conversational PDF Self-RAG Chatbot")
st.caption("Streaming FastAPI Backend + Streamlit Frontend")

if not GROQ_API_KEY:
    st.error("Missing Groq API Key. Please set GROQ_API_KEY in your .env file.")
    st.stop()

if not GOOGLE_API_KEY:
    st.warning("Missing Google API Key. Please set GOOGLE_API_KEY in your .env file for native Google Gemini Embeddings.")

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "messages_by_session" not in st.session_state:
    st.session_state.messages_by_session = {}

if "uploaded_hash" not in st.session_state:
    st.session_state.uploaded_hash = None

if "is_ready" not in st.session_state:
    st.session_state.is_ready = False

# Use a default session ID since the sidebar is removed
session_id = "default_session"

if session_id not in st.session_state.messages_by_session:
    st.session_state.messages_by_session[session_id] = []

session_messages = st.session_state.messages_by_session[session_id]

# ---------------------------------------------------------
# PDF Upload & Ingestion Pipeline (via FastAPI)
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
        with st.spinner("Uploading and indexing PDFs on backend server..."):
            try:
                # Prepare files for multipart/form-data upload
                files_payload = [
                    ("files", (f.name, f.read(), "application/pdf")) for f in uploaded_files
                ]
                
                response = requests.post(f"{API_URL}/upload", files=files_payload)
                response.raise_for_status()
                
                st.session_state.is_ready = True
                st.session_state.uploaded_hash = current_hash
                st.success(f"Processed {len(uploaded_files)} PDF(s) into {response.json().get('chunks_indexed', 0)} indexed chunks.")
            except Exception as e:
                st.error(f"Failed to connect to FastAPI backend: {str(e)}")

st.divider()

# ---------------------------------------------------------
# Conversation View & Query Execution (via SSE Streaming)
# ---------------------------------------------------------
render_chat_history(session_messages)

user_query = st.chat_input("Ask a question about your PDF documents...")

if user_query:
    if not st.session_state.is_ready:
        st.warning("Please upload at least one PDF document before asking questions.")
    else:
        # Append and display user turn
        session_messages.append({"role": "user", "content": user_query})
        render_message_turn(role="user", content=user_query)

        # Execute query via FastAPI Streaming Endpoint
        with st.chat_message("assistant"):
            try:
                # Connect to FastAPI SSE stream
                response = requests.post(
                    f"{API_URL}/chat/stream",
                    data={"query": user_query, "session_id": session_id},
                    stream=True
                )
                response.raise_for_status()

                answer_placeholder = st.empty()
                full_answer = ""
                final_sources = []
                
                # Iterate over Server-Sent Events (SSE) stream
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data: "):
                            data_str = decoded_line[6:]
                            
                            if data_str == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(data_str)
                                event_type = data.get("type")
                                
                                # Stream LLM Token
                                if event_type == "token":
                                    full_answer += data["content"]
                                    answer_placeholder.markdown(full_answer + "▌")
                                    
                                # Receive final sources at the end
                                elif event_type == "sources":
                                    final_sources = data.get("content", [])
                            except json.JSONDecodeError:
                                continue

                # Finalize answer display
                answer_placeholder.markdown(full_answer)

                # Render sources expander
                if final_sources:
                    with st.expander("Retrieved Source Passages", expanded=False):
                        for s in final_sources:
                            st.markdown(f"**{s['source']} (Page {s['page']})**")
                            st.markdown(f"> {s['snippet']}")

                # Persist assistant turn in session
                session_messages.append({
                    "role": "assistant",
                    "content": full_answer,
                    "reflections": [],
                    "sources": final_sources
                })

            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to FastAPI backend. Is the server running? (`uvicorn api:app --reload`)")
            except Exception as e:
                st.error(f"Error during execution: {str(e)}")