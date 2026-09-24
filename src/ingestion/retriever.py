import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

from src.config import (
    RETRIEVER_K,
    BM25_WEIGHT,
    SEMANTIC_WEIGHT,
    get_embeddings
)

logger = logging.getLogger(__name__)

def build_semantic_retriever(splits: List[Document]) -> BaseRetriever:
    """Build dense vector retriever backed by ChromaDB and Google Gemini embeddings."""
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

def build_bm25_retriever(splits: List[Document]) -> BaseRetriever:
    """Build sparse lexical retriever using BM25 keyword matching."""
    bm25 = BM25Retriever.from_documents(splits)
    bm25.k = RETRIEVER_K
    return bm25

def build_hybrid_retriever(splits: List[Document]) -> BaseRetriever:
    """
    Construct a hybrid EnsembleRetriever blending sparse BM25 and dense semantic
    retrievers using Reciprocal Rank Fusion (RRF).
    """
    dense_retriever = build_semantic_retriever(splits)
    sparse_retriever = build_bm25_retriever(splits)

    return EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=[BM25_WEIGHT, SEMANTIC_WEIGHT]
    )
