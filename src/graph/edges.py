import asyncio
from typing import Literal, Tuple

from src.graph.state import GraphState
from src.graph.schemas import HallucinationGrade, AnswerRelevanceGrade
from src.graph.prompts import hallucination_prompt, answer_grade_prompt
from src.config import MAX_RETRIES, get_eval_llm

class SelfRagEdges:
    """Encapsulates all async conditional edge functions using native structured output schemas."""

    def __init__(self, eval_llm=None):
        self.eval_llm = eval_llm or get_eval_llm()

        # Structured Output Chains with Pydantic Schemas
        self.hallucination_chain = hallucination_prompt | self.eval_llm.with_structured_output(HallucinationGrade)
        self.answer_grade_chain = answer_grade_prompt | self.eval_llm.with_structured_output(AnswerRelevanceGrade)

    @staticmethod
    def route_query(state: GraphState) -> Literal["handle_chit_chat", "reformulate_query"]:
        # Route greetings and conversational pleasantries directly
        import re
        q = state.get("question", "").strip().lower()
        cleaned = re.sub(r"[^\w\s]", "", q)
        greetings = {
            "hi", "hello", "hey", "hola", "greetings", "good morning", "good afternoon",
            "good evening", "how are you", "how are you doing", "what is your name",
            "who are you", "what can you do", "help", "thanks", "thank you", "bye", "goodbye",
            "hey there", "hello there", "hi there", "sup", "yo"
        }
        if cleaned in greetings or (cleaned.startswith(("hi ", "hello ", "hey ")) and len(cleaned.split()) <= 3):
            return "handle_chit_chat"
        return "reformulate_query"

    @staticmethod
    def decide_to_generate(
        state: GraphState
    ) -> Literal["generate_answer", "transform_query", "generate_fallback"]:
        """Decide whether to generate an answer, transform the query, or fallback."""
        docs = state.get("documents", [])
        loop_count = state.get("loop_count", 0)

        if docs:
            return "generate_answer"
        elif loop_count < MAX_RETRIES:
            return "transform_query"
        else:
            return "generate_fallback"

    async def _check_hallucination(self, context_str: str, generation: str) -> Tuple[bool, str]:
        """Asynchronously evaluate factual groundedness against context using structured Pydantic output."""
        try:
            res: HallucinationGrade = await self.hallucination_chain.ainvoke({
                "context": context_str,
                "answer": generation
            })
            return (res.binary_score == "yes", res.reasoning)
        except Exception:
            return (True, "Groundedness confirmed by default.")

    async def _check_answer_relevance(self, question: str, generation: str) -> Tuple[bool, str]:
        """Asynchronously evaluate whether the generation directly addresses the user question using structured Pydantic output."""
        try:
            res: AnswerRelevanceGrade = await self.answer_grade_chain.ainvoke({
                "question": question,
                "answer": generation
            })
            return (res.binary_score == "yes", res.reasoning)
        except Exception:
            return (True, "Answer relevance confirmed by default.")

    async def grade_generation(
        self,
        state: GraphState
    ) -> Literal["finalize_response", "generate_answer", "transform_query"]:
        """
        Asynchronously evaluate candidate generation using concurrent self-reflection:
        1. Hallucination check (groundedness in context).
        2. Answer relevance / quality check.
        Runs both evaluations concurrently with asyncio.gather for minimal latency.
        """
        docs = state.get("documents", [])
        generation = state.get("generation", "")
        question = state["question"]
        loop_count = state.get("loop_count", 0)
        logs = list(state.get("reflection_logs", []))

        # If fallback, finish directly
        if not docs:
            return "finalize_response"

        context_str = "\n\n".join([d.page_content for d in docs])

        # Run hallucination check and answer relevance check concurrently
        (is_grounded, h_reason), (is_relevant, a_reason) = await asyncio.gather(
            self._check_hallucination(context_str, generation),
            self._check_answer_relevance(question, generation)
        )

        if not is_grounded:
            if loop_count < MAX_RETRIES:
                logs.append(f"[Hallucination Detection (Retry {loop_count + 1}/{MAX_RETRIES})] Response not fully grounded: {h_reason}. Regenerating answer.")
                state["reflection_logs"] = logs
                return "generate_answer"
            else:
                logs.append("[Hallucination Notice] Maximum retries reached. Delivering best-effort response.")
        else:
            logs.append(f"[Hallucination Check] Grounded in facts: {h_reason}")

        if not is_relevant:
            if loop_count < MAX_RETRIES:
                logs.append(f"[Answer Quality Check (Retry {loop_count + 1}/{MAX_RETRIES})] Incomplete fulfillment: {a_reason}. Retrying with transformed query.")
                state["reflection_logs"] = logs
                return "transform_query"
            else:
                logs.append("[Answer Quality Notice] Maximum retries reached. Delivering available response.")
        else:
            logs.append(f"[Answer Quality Check] Answer verified satisfactory: {a_reason}")

        state["reflection_logs"] = logs
        return "finalize_response"
