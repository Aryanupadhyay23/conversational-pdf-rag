import streamlit as st
from src.config import GEMINI_EMBEDDING_MODEL, EMBEDDING_DIMENSION
from src.core.service import RagService

def render_sidebar(rag_service: RagService) -> str:
    """
    Render sidebar configuration controls, session switcher, and architecture summaries.
    
    Returns:
        str: Active conversation session identifier.
    """
    with st.sidebar:
        st.header("Configuration")
        session_id = st.text_input(
            "Conversation Session ID",
            value="default_session",
            help="Change this to switch between isolated conversation memory threads."
        )

        st.markdown("---")
        st.subheader("Conversational Memory")
        st.markdown("""
        * **Thread Persistence:** Chat turns are tracked by LangGraph checkpointer.
        * **Contextual Retrieval:** Queries are resolved using previous memory turns.
        * **Multi-Session Isolation:** Switch sessions to load independent memory threads.
        """)

        st.markdown("---")
        st.subheader("Hybrid Retrieval")
        st.markdown(f"""
        * **BM25 Lexical:** Sparse keyword and exact term matching.
        * **Gemini Semantic:** Google Gemini (`{GEMINI_EMBEDDING_MODEL}`, {EMBEDDING_DIMENSION}-dim).
        * **Ensemble (RRF):** Blended reciprocal rank fusion scoring.
        """)

        st.markdown("---")
        st.subheader("Async Engine")
        st.markdown("""
        * **Non-blocking Nodes:** Native asynchronous LLM & retriever operations.
        * **Parallel Grading:** Chunks evaluated concurrently via `asyncio.gather`.
        * **Concurrent Reflection:** Hallucination and relevance checks run in parallel.
        """)

        if st.button("Clear Conversation Memory", use_container_width=True):
            if "messages_by_session" in st.session_state and session_id in st.session_state.messages_by_session:
                st.session_state.messages_by_session[session_id] = []
            rag_service.clear_memory()
            st.success("Memory cleared for this session!")
            st.rerun()

    return session_id
