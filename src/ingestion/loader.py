import os
import tempfile
import logging
from typing import List, Sequence
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

logger = logging.getLogger(__name__)

def load_pdf_from_path(file_path: str, filename: str) -> List[Document]:
    """
    Load PDF using UnstructuredPDFLoader in elements mode to preserve document structure
    (Titles, Headings, Paragraphs, Lists, Tables), and attach metadata.
    """
    try:
        # Load PDF using mode="elements"
        try:
            loader = UnstructuredPDFLoader(
                file_path,
                mode="elements",
                strategy="fast"
            )
            raw_docs = loader.load()
        except Exception:
            loader = UnstructuredPDFLoader(
                file_path,
                mode="elements"
            )
            raw_docs = loader.load()
    except Exception as e:
        logger.error(f"Error loading PDF '{filename}' with UnstructuredPDFLoader: {e}")
        return []

    # Preserve Unstructured metadata and attach application metadata (source_name, page)
    valid_docs = []
    for doc in raw_docs:
        # Filter out empty or whitespace-only elements
        if not doc.page_content or not doc.page_content.strip():
            continue

        doc.metadata["source_name"] = filename
        if "page_number" in doc.metadata:
            doc.metadata["page"] = doc.metadata["page_number"] - 1
        elif "page" not in doc.metadata:
            doc.metadata["page"] = 0

        valid_docs.append(doc)

    # Filter complex metadata structures (e.g. coordinates dict) for vector-store compatibility
    return filter_complex_metadata(valid_docs)

def load_pdf_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Persist uploaded PDF bytes to a secure temporary file and load using UnstructuredPDFLoader.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_pdf_path = temp_file.name

    try:
        return load_pdf_from_path(temp_pdf_path, filename)
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

import concurrent.futures

def _process_single_file(file_bytes: bytes, filename: str) -> List[Document]:
    """Helper method for concurrent processing."""
    return load_pdf_from_bytes(file_bytes, filename)

def load_uploaded_pdfs(uploaded_files: Sequence) -> List[Document]:
    """Load and extract structured elements from an iterable of uploaded file objects concurrently."""
    all_documents = []
    
    # Use ThreadPoolExecutor for concurrent multi-PDF parsing to dramatically speed up ingestion
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_process_single_file, f.read(), f.name) 
            for f in uploaded_files
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                docs = future.result()
                all_documents.extend(docs)
            except Exception as e:
                logger.error(f"Error processing PDF in executor: {e}")
                
    return all_documents
