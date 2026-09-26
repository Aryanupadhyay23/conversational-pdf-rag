import os
import tempfile
import logging
from typing import List, Sequence
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

logger = logging.getLogger(__name__)

def load_pdf_from_path(file_path: str, filename: str) -> List[Document]:
    # Extract documents from PDF using PyPDFLoader with fallback to pypdf.PdfReader
    raw_docs = []
    try:
        loader = PyPDFLoader(file_path)
        raw_docs = loader.load()
    except Exception as e:
        logger.warning(f"PyPDFLoader failed for '{filename}': {e}. Attempting direct pypdf fallback.")
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    raw_docs.append(Document(
                        page_content=text,
                        metadata={"source_name": filename, "page": page_num}
                    ))
        except Exception as fallback_err:
            logger.error(f"Fallback extraction also failed for '{filename}': {fallback_err}")
            return []

    valid_docs = []
    for doc in raw_docs:
        if not doc.page_content or not doc.page_content.strip():
            continue

        doc.metadata["source_name"] = filename
        if "page" not in doc.metadata:
            if "page_number" in doc.metadata:
                doc.metadata["page"] = doc.metadata["page_number"] - 1
            else:
                doc.metadata["page"] = 0

        valid_docs.append(doc)

    return filter_complex_metadata(valid_docs)

def load_pdf_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    # Persist uploaded PDF bytes to a temporary file and load
    if not file_bytes:
        logger.warning(f"Empty byte buffer received for '{filename}'")
        return []

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_pdf_path = temp_file.name

    try:
        return load_pdf_from_path(temp_pdf_path, filename)
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

def load_uploaded_pdfs(uploaded_files: Sequence) -> List[Document]:
    # Load and extract text elements from an iterable of uploaded file objects
    all_documents = []
    for f in uploaded_files:
        try:
            if hasattr(f, "getvalue"):
                file_bytes = f.getvalue()
            else:
                f.seek(0)
                file_bytes = f.read()

            filename = getattr(f, "name", "document.pdf")
            docs = load_pdf_from_bytes(file_bytes, filename)
            all_documents.extend(docs)
        except Exception as e:
            logger.error(f"Error processing uploaded PDF '{getattr(f, 'name', 'unknown')}': {e}")

    return all_documents

