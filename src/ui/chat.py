from typing import List, Dict, Any
import streamlit as st

def render_message_turn(
    role: str,
    content: str,
    reflections: List[str] = None,
    sources: List[Dict[str, Any]] = None
):
    """Render an individual chat message with optional reflection logs and source citations."""
    with st.chat_message(role):
        st.markdown(content)

def render_chat_history(session_messages: List[Dict[str, Any]]):
    """Render all historical turns in the current session."""
    for msg in session_messages:
        render_message_turn(
            role=msg["role"],
            content=msg["content"],
            reflections=msg.get("reflections"),
            sources=msg.get("sources")
        )
