from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

import uuid

def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Return configured RecursiveCharacterTextSplitter instance."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

def split_documents(documents: List[Document]) -> List[Document]:
    """Split raw loaded documents into semantically coherent chunks."""
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    
    # Assign a unique ID to each chunk for precise deduplication during hybrid retrieval
    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        
    return chunks
