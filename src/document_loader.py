"""
Backwards-compatible module re-exporting document ingestion pipeline.
Direct imports from `src.ingestion` are recommended for modular use.
"""
from src.ingestion.pipeline import ingest_documents

# Backwards compatibility alias
process_and_index_pdfs = ingest_documents

__all__ = ["process_and_index_pdfs", "ingest_documents"]
