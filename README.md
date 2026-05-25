# Conversational PDF RAG Chatbot

**Live Demo:** [Hugging Face Spaces: Conversational_PDF_RAG](https://huggingface.co/spaces/Aryan2301/Conversational_PDF_RAG)

An end-to-end Conversational Retrieval-Augmented Generation (RAG) application built using Streamlit, LangChain, Groq, ChromaDB, and Hugging Face embeddings. This application allows users to upload multiple PDF documents and interact with them through a conversational AI interface with contextual memory and history-aware retrieval.

The chatbot intelligently maintains chat history and reformulates follow-up questions using contextual retrieval to ensure accurate, context-aware answers.

---

## Features

* **Multiple PDF Uploads:** Process and query several documents simultaneously via a temporary PDF processing pipeline.
* **Conversational Interface with Streaming:** Real-time, typewriter-effect responses via LangChain's streaming capabilities.
* **Session-Based Memory:** Manage different conversations effortlessly using customizable Session IDs.
* **Advanced RAG Pipeline:** Context-aware follow-up question handling and standalone query reformulation.
* **High-Performance LLM:** Powered by Groq's `llama-3.3-70b-versatile` model.
* **Robust Embeddings & Storage:** Utilizes Hugging Face's `all-MiniLM-L6-v2` embeddings stored locally in ChromaDB.
* **Docker Ready:** Fully containerized for quick and consistent deployments.

---

## Tech Stack

* **Backend & Frameworks:** Python, Streamlit, LangChain (`langchain-core`, `langchain-classic`, `langchain-community`)
* **LLM Provider:** Groq
* **Vector Database:** ChromaDB
* **Embeddings:** Hugging Face / Sentence Transformers (`all-MiniLM-L6-v2`)
* **Document Processing:** PyPDFLoader, RecursiveCharacterTextSplitter

---

## Application Workflow

1. **Ingestion:** Upload one or more PDF files.
2. **Extraction:** Extract document text.
3. **Processing:** Split text into manageable chunks (chunk_size = 5000, chunk_overlap = 500).
4. **Embedding:** Generate embeddings for text chunks.
5. **Storage:** Store embeddings in ChromaDB.
6. **Query Reformulation:** When a follow-up question is asked, the system uses the chat history to reformulate it into a standalone search query.
    * *Example User Input:* "What is CNN?" -> "How does it work?"
    * *Internally Reformulated Query:* "How does Convolutional Neural Network (CNN) work?"
7. **Retrieval:** Retrieve relevant chunks from the vector database.
8. **Generation:** Generate contextual response using the Groq LLM and maintain conversational memory.

---

## Getting Started

### Prerequisites

* Python 3.11+
* [Groq API Key](https://console.groq.com/keys)
* [Hugging Face Token](https://huggingface.co/settings/tokens)

### 1. Clone the Repository

    git clone <your_repo_url>
    cd conversational-pdf-rag

### 2. Set Up Environment Variables

Create a `.env` file in the root directory and add your API keys:

    GROQ_API_KEY=your_groq_api_key
    HF_TOKEN=your_huggingface_token

---

### Option A: Local Installation

1. **Install dependencies:**

        pip install -r requirements.txt

2. **Run the application:**

        streamlit run app.py

   The application will start locally on `http://localhost:8501`.

### Option B: Docker Deployment

1. **Build the Docker image:**

        docker build -t conversational-pdf-rag .

2. **Run the Docker container:**
   *(Ensure your `.env` file is in the same directory, or pass variables directly)*

        docker run -p 8501:8501 --env-file .env conversational-pdf-rag

---

## Project Structure

    conversational-pdf-rag/
    │
    ├── app.py               # Main Streamlit application and RAG pipeline
    ├── requirements.txt     # Python dependencies
    ├── Dockerfile           # Docker configuration for containerization
    ├── .env                 # Environment variables (API keys)
    ├── .gitignore           # Git ignore rules
    └── README.md            # Project documentation

---

## Example Use Cases

* Research paper Q&A
* Technical documentation assistant
* Educational PDF chatbot
* Resume/document analysis
* Multi-document conversational search
* Notes and ebook interaction

---

## Future Improvements

* Persistent vector database storage
* Source citations with page numbers
* Hybrid search (BM25 + Vector Search)
* Authentication system
* Cloud deployment support
* File management dashboard
* Multi-user support

---

## License

This project is open-source and available under the MIT License.