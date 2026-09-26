from typing import Literal
from pydantic import BaseModel, Field, field_validator

def _normalize_binary(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return "yes" if v == 1 else "no"
    if isinstance(v, str):
        clean = v.strip().lower()
        if clean in ("yes", "y", "true", "1"):
            return "yes"
        return "no"
    return "no"

class DocumentRelevanceGrade(BaseModel):
    """Structured evaluation of document chunk relevance to the search query."""
    binary_score: Literal["yes", "no"] = Field(
        description="Binary relevance score: 'yes' if document contains facts/concepts relevant to the query, 'no' otherwise."
    )
    reasoning: str = Field(
        description="Concise rationale explaining the grading decision."
    )

    @field_validator("binary_score", mode="before")
    @classmethod
    def validate_binary_score(cls, v):
        return _normalize_binary(v)

class HallucinationGrade(BaseModel):
    """Structured evaluation of factual groundedness of an answer against reference context."""
    binary_score: Literal["yes", "no"] = Field(
        description="Binary groundedness score: 'yes' if the answer is strictly supported by context, 'no' if it introduces hallucinations."
    )
    reasoning: str = Field(
        description="Concise explanation of whether statements in the answer are grounded in context facts."
    )

    @field_validator("binary_score", mode="before")
    @classmethod
    def validate_binary_score(cls, v):
        return _normalize_binary(v)

class AnswerRelevanceGrade(BaseModel):
    """Structured evaluation of whether the answer directly resolves the user question."""
    binary_score: Literal["yes", "no"] = Field(
        description="Binary relevance score: 'yes' if answer directly and adequately addresses the question, 'no' otherwise."
    )
    reasoning: str = Field(
        description="Concise explanation of how well the answer addresses the user's inquiry."
    )

    @field_validator("binary_score", mode="before")
    @classmethod
    def validate_binary_score(cls, v):
        return _normalize_binary(v)

class ReformulatedQuery(BaseModel):
    """Structured standalone search query contextualized using conversational history."""
    standalone_query: str = Field(
        description="Self-contained search query optimized for retrieval, resolving all pronouns and conversational references."
    )
    reasoning: str = Field(
        description="Brief justification for how references were resolved or why the query was kept as-is."
    )

class TransformedQuery(BaseModel):
    """Structured rewritten query designed to overcome failed retrieval attempts."""
    transformed_query: str = Field(
        description="Optimized search query rephrased with alternative technical keywords, synonyms, and expanded terminology."
    )
    search_strategy: str = Field(
        description="Brief explanation of the optimization strategy applied to improve recall."
    )
