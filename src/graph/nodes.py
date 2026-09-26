import asyncio
from typing import Optional, List
from langchain_core.messages import AIMessage, filter_messages
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.graph.state import GraphState
from src.graph.schemas import (
    DocumentRelevanceGrade,
    ReformulatedQuery,
    TransformedQuery
)
from src.graph.prompts import (
    reformulate_prompt,
    doc_grade_prompt,
    rag_prompt,
    transform_query_prompt
)
from src.config import MAX_RETRIES, get_llm, get_eval_llm
from src.utils.helpers import format_docs_to_context

class SelfRagNodes:
    """Encapsulates all async node functions for Self-RAG using native LangChain LCEL & structured output chains."""

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        llm = None,
        eval_llm = None
    ):
        self.retriever = retriever
        self.llm = llm or get_llm()
        self.eval_llm = eval_llm or get_eval_llm()

        # Structured Output Chains with Pydantic Schemas
        self.reformulate_chain = reformulate_prompt | self.eval_llm.with_structured_output(ReformulatedQuery)
        self.doc_grade_chain = doc_grade_prompt | self.eval_llm.with_structured_output(DocumentRelevanceGrade)
        self.transform_query_chain = transform_query_prompt | self.eval_llm.with_structured_output(TransformedQuery)

        # Standard RAG Generation Chain
        self.rag_chain = rag_prompt | self.llm | StrOutputParser()

    async def reformulate_query(self, state: GraphState) -> dict:
        """
        Asynchronously reformulate follow-up user query into standalone search query
        using conversational history and structured output.
        """
        question = state["question"]
        messages = state["messages"]
        logs = list(state.get("reflection_logs", []))

        # LangChain native message filtering for chat history (limited to last 20 messages)
        chat_history = filter_messages(messages, include_types=["human", "ai"])[:-1][-20:]

        if chat_history:
            try:
                res: ReformulatedQuery = await self.reformulate_chain.ainvoke({
                    "chat_history": chat_history,
                    "question": question
                })
                standalone = res.standalone_query.strip()
                logs.append(f"[Query Reformulation] Contextualized query: `{standalone}` ({res.reasoning})")
            except Exception:
                standalone = question
                logs.append(f"[Query Formulation] Using direct query: `{standalone}`")
        else:
            standalone = question
            logs.append(f"[Query Formulation] Using direct query: `{standalone}`")

        return {"standalone_query": standalone, "reflection_logs": logs}

    async def retrieve_documents(self, state: GraphState) -> dict:
        """Asynchronously fetch candidate document chunks from hybrid retriever."""
        query = state["standalone_query"]
        logs = list(state.get("reflection_logs", []))

        if self.retriever is None:
            docs = []
        else:
            # LangChain native retriever ainvoke
            docs = await self.retriever.ainvoke(query)

        logs.append(f"[Hybrid Retrieval] Retrieved {len(docs)} candidate chunks using BM25 + Chroma Semantic Search (EnsembleRetriever).")
        return {"documents": docs, "reflection_logs": logs}

    async def _grade_single_doc(self, doc: Document, query: str) -> Optional[Document]:
        """Asynchronously grade an individual document chunk using structured Pydantic scoring."""
        try:
            grade: DocumentRelevanceGrade = await self.doc_grade_chain.ainvoke({
                "query": query,
                "document": doc.page_content
            })
            if grade.binary_score == "yes":
                return doc
        except Exception:
            # On parsing/API anomaly, retain document to avoid dropping relevant context
            return doc
        return None

    async def grade_documents(self, state: GraphState) -> dict:
        """
        Assess relevance of all retrieved chunks concurrently using asyncio.gather.
        Filters out non-relevant chunks in parallel for maximum speed.
        """
        query = state["standalone_query"]
        docs = state.get("documents", [])
        logs = list(state.get("reflection_logs", []))

        # Parallel document grading across all retrieved chunks
        grading_tasks = [self._grade_single_doc(doc, query) for doc in docs]
        graded_results = await asyncio.gather(*grading_tasks)

        relevant_docs = [doc for doc in graded_results if doc is not None]

        logs.append(f"[Document Relevance Grading] Retained {len(relevant_docs)} of {len(docs)} relevant candidate chunks.")
        return {"documents": relevant_docs, "reflection_logs": logs}

    async def transform_query(self, state: GraphState) -> dict:
        """Asynchronously rewrite search query using structured query optimizer."""
        query = state["standalone_query"]
        loop_count = state.get("loop_count", 0) + 1
        logs = list(state.get("reflection_logs", []))

        try:
            res: TransformedQuery = await self.transform_query_chain.ainvoke({"query": query})
            new_query = res.transformed_query.strip()
            logs.append(f"[Query Transformation (Attempt {loop_count}/{MAX_RETRIES})] Rewriting query to `{new_query}` ({res.search_strategy}).")
        except Exception:
            new_query = query
            logs.append(f"[Query Transformation (Attempt {loop_count}/{MAX_RETRIES})] Reusing query `{new_query}`.")

        return {
            "standalone_query": new_query,
            "loop_count": loop_count,
            "reflection_logs": logs
        }

    async def generate_answer(self, state: GraphState) -> dict:
        """
        Asynchronously synthesize answer using retrieved relevant documents
        and conversation history filtered natively via LangChain filter_messages.
        """
        docs = state.get("documents", [])
        question = state["question"]
        messages = state["messages"]
        logs = list(state.get("reflection_logs", []))

        context_str = format_docs_to_context(docs)
        # Limit chat history to the last 20 messages
        chat_history = filter_messages(messages, include_types=["human", "ai"])[:-1][-20:]

        generation = await self.rag_chain.ainvoke({
            "context": context_str,
            "chat_history": chat_history,
            "question": question
        })

        loop_count = state.get("loop_count", 0)
        if state.get("generation"):
            loop_count += 1

        logs.append(f"[Answer Generation] Generated candidate response incorporating {len(chat_history)} memory messages.")
        return {
            "generation": generation,
            "loop_count": loop_count,
            "reflection_logs": logs
        }

    async def handle_chit_chat(self, state: GraphState) -> dict:
        # Handle general greetings and conversational chit-chat directly
        question = state["question"]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a warm, helpful AI assistant specializing in answering questions about uploaded PDF documents. Reply politely and warmly to the user's greeting or remark, and invite them to ask any questions about the documents they have uploaded. Keep your reply concise (1-2 sentences)."),
            ("human", "{question}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        try:
            reply = await chain.ainvoke({"question": question})
        except Exception:
            reply = "Hello! I am ready to help you with your uploaded PDF documents. What would you like to know?"

        return {
            "generation": reply,
            "documents": [],
            "reflection_logs": ["[Conversational] Handled greeting directly."]
        }

    async def generate_fallback(self, state: GraphState) -> dict:
        # Gracefully handle queries where no document passages matched
        question = state["question"]
        logs = list(state.get("reflection_logs", []))

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an AI assistant for uploaded PDF documents. No matching passages were found in the uploaded documents for the user's message.\n"
                "- If the user's message is a greeting, polite pleasantry, or conversational question, respond warmly and invite them to ask about their documents.\n"
                "- Otherwise, politely inform the user that the uploaded documents do not contain information on this topic and invite them to ask a question related to their documents.\n"
                "Keep your reply friendly and concise."
            )),
            ("human", "{question}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        try:
            fallback_text = await chain.ainvoke({"question": question})
        except Exception:
            fallback_text = (
                "I could not find relevant information in the uploaded PDF document(s) "
                "to answer your question. Please feel free to ask a question related to your documents or try rephrasing."
            )
        logs.append("[Fallback Triggered] Generated contextual fallback response.")
        return {"generation": fallback_text, "reflection_logs": logs}

    async def finalize_response(self, state: GraphState) -> dict:
        """Append finalized response to state messages, saving it into checkpointer memory."""
        generation = state["generation"]
        logs = list(state.get("reflection_logs", []))
        logs.append("[Completed] Final response saved into conversation memory.")
        return {
            "messages": [AIMessage(content=generation)],
            "reflection_logs": logs
        }
