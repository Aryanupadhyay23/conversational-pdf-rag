from typing import Sequence, Tuple
from langchain_core.retrievers import BaseRetriever

from src.ingestion.loader import load_uploaded_pdfs
from src.ingestion.splitter import split_documents
from src.ingestion.retriever import build_hybrid_retriever

def ingest_documents(uploaded_files: Sequence) -> Tuple[BaseRetriever, int]:
    """
    End-to-end ingestion pipeline:
    1. Loads PDF bytes with PyPDFLoader / pypdf.
    2. Splits pages with RecursiveCharacterTextSplitter while carrying metadata forward.
    3. Builds the hybrid EnsembleRetriever (BM25 + Chroma Semantic).

    Returns:
        Tuple[BaseRetriever, int]: The configured hybrid retriever and total chunk count.
    """
    raw_docs = load_uploaded_pdfs(uploaded_files)
    if not raw_docs:
        raise ValueError("No text elements could be extracted from the uploaded PDF document(s).")

    splits = split_documents(raw_docs)
    if not splits:
        raise ValueError("No valid document chunks generated from extracted PDF elements.")

    retriever = build_hybrid_retriever(splits)
    return retriever, len(splits)
