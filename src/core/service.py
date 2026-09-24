from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.memory import MemorySaver

from src.graph.workflow import build_self_rag_graph
from src.utils.helpers import run_async_safe, extract_sources_metadata

@dataclass
class RagQueryResult:
    """Standardized response from the Self-RAG service."""
    answer: str
    reflection_logs: List[str] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)

class RagService:
    """
    Core business service managing LangGraph execution, conversational checkpointer
    memory, and retrieval orchestration.
    """

    def __init__(self, retriever: Optional[BaseRetriever] = None):
        self.retriever = retriever
        self.checkpointer = MemorySaver()
        self.rag_app = None

        if self.retriever:
            self._compile_graph()

    def _compile_graph(self):
        """Compile or re-compile the Self-RAG state graph."""
        self.rag_app = build_self_rag_graph(
            retriever=self.retriever,
            checkpointer=self.checkpointer
        )

    def set_retriever(self, retriever: BaseRetriever):
        """Update active document retriever and recompile the graph."""
        self.retriever = retriever
        self._compile_graph()

    def clear_memory(self):
        """Reset conversation memory checkpointer while preserving current retriever."""
        self.checkpointer = MemorySaver()
        if self.retriever:
            self._compile_graph()

    def is_ready(self) -> bool:
        """Check if retriever and workflow are initialized."""
        return self.retriever is not None and self.rag_app is not None

    def execute_query(self, user_query: str, session_id: str = "default_session") -> RagQueryResult:
        """
        Execute an end-to-end query turn through the compiled Self-RAG graph:
        1. Configures session thread for checkpointer memory.
        2. Dispatches asynchronous execution safely.
        3. Formats generated answer, reflection trace, and source citations.
        """
        if not self.is_ready():
            raise RuntimeError("RagService is not ready. Please ingest documents first.")

        config = RunnableConfig(configurable={"thread_id": session_id})
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "question": user_query,
            "standalone_query": user_query,
            "documents": [],
            "generation": "",
            "loop_count": 0,
            "reflection_logs": []
        }

        # Safely invoke async state graph in any thread/runtime context
        final_state = run_async_safe(
            self.rag_app.ainvoke(initial_state, config=config)
        )

        answer = final_state.get("generation", "No response generated.")
        reflection_logs = final_state.get("reflection_logs", [])
        sources = extract_sources_metadata(final_state.get("documents", []))

        return RagQueryResult(
            answer=answer,
            reflection_logs=reflection_logs,
            sources=sources
        )
