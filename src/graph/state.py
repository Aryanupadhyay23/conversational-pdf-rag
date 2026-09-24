from typing import Annotated, List, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """
    Represents the state of the Self-RAG state graph.

    Attributes:
        messages: Conversation chat history, automatically appended via add_messages.
        question: Original user query.
        standalone_query: Disambiguated / reformulated search query.
        documents: Retrieved and relevance-filtered documents.
        generation: Current candidate answer.
        loop_count: Number of query transformation / regeneration iterations.
        reflection_logs: Live trace of reasoning and grading steps.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    standalone_query: str
    documents: List[Document]
    generation: str
    loop_count: int
    reflection_logs: List[str]
