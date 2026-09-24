import asyncio
import hashlib
import concurrent.futures
from typing import List, Dict, Any, Coroutine, TypeVar
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, format_document

T = TypeVar("T")

# LangChain native document prompt template
DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[Source: {source_name}, Page {page}]\n{page_content}"
)

def compute_files_hash(files) -> str:
    """Compute a combined MD5 hash of uploaded files to detect changes."""
    hasher = hashlib.md5()
    for f in files:
        hasher.update(f.name.encode())
        hasher.update(str(f.size).encode())
    return hasher.hexdigest()

def format_docs_to_context(docs: List[Document]) -> str:
    """
    Format a list of Document objects into a unified context string
    using LangChain's native format_document and PromptTemplate.
    """
    formatted_docs = []
    for doc in docs:
        doc_with_display_meta = Document(
            page_content=doc.page_content,
            metadata={
                "source_name": doc.metadata.get("source_name", "Unknown PDF"),
                "page": doc.metadata.get("page", 0) + 1
            }
        )
        formatted_docs.append(format_document(doc_with_display_meta, DOCUMENT_PROMPT))
    return "\n\n---\n\n".join(formatted_docs)

def extract_sources_metadata(docs: List[Document]) -> List[Dict[str, Any]]:
    """Extract displayable metadata and snippet from a list of Document objects."""
    sources = []
    for doc in docs:
        sources.append({
            "source": doc.metadata.get("source_name", "Unknown PDF"),
            "page": doc.metadata.get("page", 0) + 1,
            "snippet": doc.page_content[:300].replace("\n", " ") + "..."
        })
    return sources

def run_async_safe(coro: Coroutine[Any, Any, T]) -> T:
    """
    Safely execute an asynchronous coroutine in synchronous or nested environments
    (like Streamlit), avoiding 'RuntimeError: This event loop is already running'.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(coro))
            return future.result()
    else:
        return asyncio.run(coro)
