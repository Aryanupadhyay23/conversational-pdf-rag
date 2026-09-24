# Conversational PDF Self-RAG Chatbot

**Live Demo:** [Hugging Face Spaces: Conversational_PDF_RAG](https://huggingface.co/spaces/Aryan2301/Conversational_PDF_RAG)

An end-to-end **Self-Reflective Retrieval-Augmented Generation (Self-RAG)** conversational chatbot built with **LangGraph**, **Streamlit**, **Groq (Llama-3.3-70B)**, **ChromaDB**, and **Hugging Face embeddings**. This application enables users to upload multiple PDF documents and converse with their content through an autonomous, self-correcting agent state graph.

The agent actively reflects on retrieved context relevance, transforms unhelpful queries, verifies generated answers against hallucinations, and evaluates answer quality before responding.

---

## Key Features

* **Decoupled Architecture:** A lightweight Streamlit frontend client seamlessly streams Server-Sent Events (SSE) from a high-performance FastAPI backend.
* **Self-RAG Architecture with LangGraph:** Autonomous state graph workflow featuring self-reflection, hallucination detection, and query transformation loops.
* **Hybrid Retrieval (BM25 + Semantic Chroma):** Combines lexical keyword matching (BM25) and dense vector search (Chroma) fused via LangChain's native `EnsembleRetriever` using Reciprocal Rank Fusion (RRF).
* **Fully Asynchronous Execution:** Entire graph is implemented with non-blocking async nodes (`ainvoke`) and concurrent evaluations (`asyncio.gather`) for parallel document grading and simultaneous hallucination/relevance verification.
* **Conversational Checkpointer Memory:** Seamlessly preserves conversation history and context across multi-turn interactions using LangGraph's native `MemorySaver` (`thread_id=session_id`).
* **Document Relevance Reflection:** Filters out low-quality/irrelevant retrieved chunks concurrently before generation.
* **Dynamic Query Transformation:** Automatically rewrites the query using memory context and retries retrieval if initial documents are insufficient or irrelevant.
* **Hallucination & Faithfulness Guard:** Evaluates if generated responses are factually grounded in the provided document context.
* **Answer Quality Assessment:** Checks that the final answer directly and comprehensively addresses the user's inquiry.
* **Interactive Reflection Trace:** Real-time visibility into the Self-RAG decision process in Streamlit.
* **Source Passages & Citations:** Expandable view of retrieved document snippets with source file names and page references.
* **Smart Ingestion Caching:** Vectorstore indexing is cached so PDFs are only processed once per upload batch, eliminating rerun latency.
* **High-Performance LLM:** Powered by Groq's high-throughput LLM models.
* **Parallel Multi-PDF Processing:** Concurrent multi-threaded document parsing for massive speedups when uploading multiple files.
* **Observability with LangSmith:** Full execution tracing out-of-the-box to visualize the agent's graph operations, chunk retrievals, and evaluation steps.
* **Docker Ready:** Fully containerized for easy and consistent deployments.

---

## Tech Stack

* **Orchestration & Workflow:** [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain Core](https://github.com/langchain-ai/langchain)
* **Backend API:** FastAPI, Uvicorn
* **Frontend:** Streamlit
* **LLM Provider:** Groq
* **Hybrid Retrieval:** BM25 (`rank_bm25`), ChromaDB (`langchain-chroma`), LangChain `EnsembleRetriever`
* **Embeddings:** Google Gemini Embeddings (`gemini-embedding-2`, 768 dimensions via `langchain-google-genai`)
* **Document Processing:** Unstructured (`UnstructuredPDFLoader`), RecursiveCharacterTextSplitter (with UUID chunk deduplication)
* **Observability:** LangSmith

---

## Self-RAG Architecture & Workflow

```mermaid
graph TD
    Start([User Input]) --> Reformulate[Node: reformulate_query]
    Reformulate --> Retrieve[Node: retrieve_documents]
    Retrieve --> GradeDocs[Node: grade_documents]
    
    GradeDocs --> DecisionDocs{Relevant Docs Found?}
    DecisionDocs -->|Yes| Generate[Node: generate_answer]
    DecisionDocs -->|No & Retries Available| TransformQuery[Node: transform_query]
    DecisionDocs -->|No & Max Retries| GenerateFallback[Node: generate_fallback]
    
    TransformQuery --> Retrieve
    
    Generate --> GradeGeneration{Self-Reflection on Answer}
    GradeGeneration -->|Hallucination Detected & Retry Available| Generate
    GradeGeneration -->|Not Useful & Retry Available| TransformQuery
    GradeGeneration -->|Grounded & Useful| FinalOutput[Node: finalize_response]
    GenerateFallback --> FinalOutput
    
    FinalOutput --> End([Stream Response to UI & Save State])
```

### Step-by-Step Flow:
1. **Query Reformulation:** If prior chat history exists, reformulates follow-up queries into standalone search questions.
2. **Retrieval:** Fetches candidate chunks from the Chroma vector store.
3. **Document Relevance Grading:** Evaluates whether each chunk is relevant to the question; filters out noise.
4. **Adaptive Routing:** If no relevant documents are found, rewrites the search query and searches again (up to max retries). If max retries are exceeded, executes a graceful fallback.
5. **Contextual Generation:** Generates the candidate answer strictly using grounded context chunks and conversational context.
6. **Hallucination & Relevance Reflection:**
   - **Groundedness Check:** Confirms the answer contains no hallucinations.
   - **Answer Relevance Check:** Confirms the answer addresses the user's prompt.
7. **Finalization:** Updates the state graph checkpointer thread and displays the answer with full reflection telemetry and source citations.

---

## Getting Started

### Prerequisites

* Python 3.11+
* [Groq API Key](https://console.groq.com/keys)
* [Google AI Studio API Key](https://aistudio.google.com/app/apikey) (for Gemini Embeddings)
* [Hugging Face Token](https://huggingface.co/settings/tokens)

### 1. Clone the Repository

```bash
git clone <your_repo_url>
cd conversational-pdf-rag
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
HF_TOKEN=your_huggingface_token

# Optional: LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT="PDF_RAG_CHATBOT"
```

---

### Option A: Local Installation

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI Backend (Terminal 1):**

   ```bash
   uvicorn api:app --reload --port 8000
   ```

3. **Start the Streamlit Frontend (Terminal 2):**

   ```bash
   streamlit run app.py
   ```

   The application will start locally on `http://localhost:8501` and connect to the backend API.

### Option B: Docker Deployment

1. **Build the Docker image:**

   ```bash
   docker build -t conversational-pdf-self-rag .
   ```

2. **Run the Docker container:**

   ```bash
   docker run -p 8000:8000 -p 8501:8501 --env-file .env conversational-pdf-self-rag
   ```

---

## Project Structure

```
conversational-pdf-rag/
│
├── src/
│   ├── __init__.py          # Package initializer
│   ├── config.py            # Central configuration, constants, and LLM factory
│   ├── document_loader.py   # Backwards-compatible loader delegating to src.ingestion
│   │
│   ├── core/                # Core business orchestration & service layer
│   │   ├── __init__.py
│   │   └── service.py       # RagService: state graph execution, memory checkpointer, queries
│   │
│   ├── ingestion/           # Document ingestion, splitting, and hybrid retrieval
│   │   ├── __init__.py
│   │   ├── loader.py        # PDF text extraction & temporary file lifecycle
│   │   ├── splitter.py      # Semantic text splitting & chunking
│   │   ├── retriever.py     # Chroma dense & BM25 sparse hybrid retriever builders
│   │   └── pipeline.py      # End-to-end ingestion pipeline
│   │
│   ├── graph/               # Self-RAG LangGraph workflow
│   │   ├── __init__.py      # Subpackage exports
│   │   ├── state.py         # GraphState schema definition
│   │   ├── schemas.py       # Pydantic structured output models (evaluations & queries)
│   │   ├── prompts.py       # Prompt templates (reformulation, grading, generation, hallucination)
│   │   ├── nodes.py         # Self-RAG node functions (SelfRagNodes)
│   │   ├── edges.py         # Conditional routing and self-reflection logic (SelfRagEdges)
│   │   └── workflow.py      # StateGraph assembly and graph compilation
│   │
│   ├── ui/                  # Modular Streamlit UI components
│   │   ├── __init__.py
│   │   ├── sidebar.py       # Sidebar configuration, session switcher, memory controls
│   │   └── chat.py          # Message bubbles, reflection expanders, and citations
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py       # Hashing, citation formatting, safe async executor
│
├── app.py                   # Streamlit UI layer and state orchestration
├── requirements.txt         # Project dependencies (LangGraph, Groq, Chroma, Streamlit)
├── Dockerfile               # Container configuration for deployment
├── .env                     # Environment variables (API keys)
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation and Self-RAG architecture diagram
```

---

## Example Use Cases

* **Research Paper Q&A:** Deep dive into complex papers with hallucination-checked citations.
* **Technical Manuals & Documentation:** Pinpoint exact steps and verify factual accuracy.
* **Legal & Contract Analysis:** Ensure responses reflect only the uploaded text.
* **Educational Assistant:** Chat with textbooks and lecture notes with conversational memory.

---

## License

This project is open-source and available under the [MIT License](LICENSE).