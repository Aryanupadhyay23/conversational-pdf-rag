from typing import Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.retrievers import BaseRetriever
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.graph.state import GraphState
from src.graph.nodes import SelfRagNodes
from src.graph.edges import SelfRagEdges

def build_self_rag_graph(
    retriever: Optional[BaseRetriever] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    llm = None,
    eval_llm = None
):
    """
    Construct and compile the Self-RAG StateGraph.

    Args:
        retriever: Chroma vector store retriever.
        checkpointer: LangGraph checkpointer for state/session persistence.
        llm: Generation LLM instance.
        eval_llm: Evaluation/Grading LLM instance.

    Returns:
        CompiledStateGraph: The compiled executable Self-RAG graph.
    """
    nodes = SelfRagNodes(retriever=retriever, llm=llm, eval_llm=eval_llm)
    edges = SelfRagEdges(eval_llm=nodes.eval_llm)

    workflow = StateGraph(GraphState)

    # Register Nodes
    workflow.add_node("reformulate_query", nodes.reformulate_query)
    workflow.add_node("retrieve_documents", nodes.retrieve_documents)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("transform_query", nodes.transform_query)
    workflow.add_node("generate_answer", nodes.generate_answer)
    workflow.add_node("generate_fallback", nodes.generate_fallback)
    workflow.add_node("finalize_response", nodes.finalize_response)

    # Standard Edges
    workflow.add_edge(START, "reformulate_query")
    workflow.add_edge("reformulate_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "grade_documents")

    # Document Evaluation Conditional Edge
    workflow.add_conditional_edges(
        "grade_documents",
        edges.decide_to_generate,
        {
            "generate_answer": "generate_answer",
            "transform_query": "transform_query",
            "generate_fallback": "generate_fallback"
        }
    )

    # Query Transformation Loop
    workflow.add_edge("transform_query", "retrieve_documents")

    # Generation Evaluation Conditional Edge (Self-Correction Loop)
    workflow.add_conditional_edges(
        "generate_answer",
        edges.grade_generation,
        {
            "finalize_response": "finalize_response",
            "generate_answer": "generate_answer",
            "transform_query": "transform_query"
        }
    )

    workflow.add_edge("generate_fallback", "finalize_response")
    workflow.add_edge("finalize_response", END)

    return workflow.compile(checkpointer=checkpointer)
