import asyncio
import json
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from src.core.service import RagService
from src.ingestion.pipeline import ingest_documents
from src.utils.helpers import extract_sources_metadata

app = FastAPI(title="Conversational PDF Self-RAG Streaming API")

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RagService instance
rag_service = RagService()

class SyncUploadWrapper:
    """Wrapper to make FastAPI async UploadFile compatible with synchronous ingestion logic."""
    def __init__(self, name: str, content: bytes):
        self.name = name
        self.content = content

    def read(self):
        return self.content

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Async endpoint to ingest multiple PDFs concurrently."""
    try:
        wrapped_files = []
        for file in files:
            content = await file.read()
            wrapped_files.append(SyncUploadWrapper(file.filename, content))
            
        # Offload CPU-heavy ingestion to thread pool
        retriever, chunk_count = await asyncio.to_thread(ingest_documents, wrapped_files)
        rag_service.set_retriever(retriever)
        return {"status": "success", "chunks_indexed": chunk_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(query: str = Form(...), session_id: str = Form("default_session")):
    """
    Execute Self-RAG pipeline asynchronously and stream the reflection logs 
    and final generated answer using Server-Sent Events (SSE).
    """
    if not rag_service.is_ready():
        raise HTTPException(status_code=400, detail="Please upload at least one PDF first.")

    async def event_generator():
        config = RunnableConfig(configurable={"thread_id": session_id})
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "question": query,
            "standalone_query": query,
            "documents": [],
            "generation": "",
            "loop_count": 0,
            "reflection_logs": []
        }

        # Natively stream graph node execution state asynchronously using v2 events
        async for event in rag_service.rag_app.astream_events(initial_state, config=config, version="v2"):
            kind = event["event"]
            
            # 1. Stream true LLM tokens exclusively from the generator node
            if kind == "on_chat_model_stream":
                metadata = event.get("metadata", {})
                if metadata.get("langgraph_node") == "generate_answer":
                    content = event["data"]["chunk"].content
                    if content:
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
            
            # 2. Provide final sources at the end of the final node
            elif kind == "on_chain_end" and event.get("name") == "finalize_response":
                state = event["data"].get("output", {})
                if state:
                    sources = extract_sources_metadata(state.get("documents", []))
                    yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"
                    
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
