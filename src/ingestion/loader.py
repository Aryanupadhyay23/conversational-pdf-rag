import os
import tempfile
import logging
from typing import List, Sequence
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

def sanitize_doc_metadata(raw_metadata: dict, filename: str) -> dict:
    """
    Ensure all metadata values are primitive types (str, int, float, bool)
    and guarantee that 'source_name' and 'page' are preserved accurately.
    """
    cleaned = {}
    for k, v in raw_metadata.items():
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v

    # Explicitly guarantee source_name is always a string containing the filename
    cleaned["source_name"] = str(filename)

    # Safely resolve page number to integer (0-indexed)
    page_val = raw_metadata.get("page")
    if page_val is None:
        page_val = raw_metadata.get("page_number")
        if page_val is not None:
            try:
                page_val = int(page_val) - 1
            except (ValueError, TypeError):
                page_val = 0
        else:
            page_val = 0
    else:
        try:
            page_val = int(page_val)
        except (ValueError, TypeError):
            page_val = 0

    cleaned["page"] = max(0, page_val)
    return cleaned

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

        doc.metadata = sanitize_doc_metadata(doc.metadata, filename)
        valid_docs.append(doc)

    return valid_docs

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

