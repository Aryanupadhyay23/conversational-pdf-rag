# Conversational PDF RAG Chatbot

Live Demo: https://huggingface.co/spaces/Aryan2301/Conversational_PDF_RAG

A conversational Retrieval Augmented Generation (RAG) application built using Streamlit, LangChain, Groq, ChromaDB, and HuggingFace embeddings. The application allows users to upload multiple PDF files and interact with their content through a conversational interface.

The chatbot maintains chat history and understands follow-up questions using contextual retrieval.

## Features

- Multiple PDF upload support
- Conversational chat interface
- Chat history memory
- Retrieval Augmented Generation (RAG)
- Context aware follow-up question handling
- ChromaDB vector storage
- HuggingFace embeddings
- Groq LLM integration
- Streamlit based interface

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq
- ChromaDB
- HuggingFace Embeddings
- Sentence Transformers

## Workflow

1. Upload one or more PDF files
2. Extract and split document text into chunks
3. Generate embeddings for document chunks
4. Store embeddings in ChromaDB
5. Retrieve relevant chunks based on user query
6. Generate contextual responses using LLM

The application also reformulates follow-up questions into standalone questions before retrieval to improve retrieval accuracy.

Example:

```text id="a8s3d7"
User: What is CNN?
User: How does it work?
```

Converted internally into:

```text id="r4t7m2"
How does Convolutional Neural Network (CNN) work?
```

## Installation

Clone the repository:

```bash id="m6x2k9"
git clone <your_repo_url>
```

Move into the project directory:

```bash id="q2n8p5"
cd conversational-pdf-rag
```

Install dependencies:

```bash id="f7v3w1"
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the root directory:

```env id="u5y1b4"
HF_TOKEN=your_huggingface_token
```

The Groq API key is entered directly through the application UI.

## Run the Application

```bash id="e3r8c6"
streamlit run app.py
```

## Requirements

```txt id="x4m1v8"
streamlit==1.57.0
python-dotenv==1.2.2

langchain==1.2.18
langchain-core==1.3.3
langchain-community==0.4.1
langchain-classic==1.0.7

langchain-groq==1.1.2
langchain-chroma==1.1.0
langchain-huggingface==1.2.2

chromadb==1.5.9
pypdf==6.11.0
sentence-transformers==5.4.1

huggingface_hub==1.14.0
groq==0.37.1
```

## Project Structure

```text id="k9t4p2"
│── app.py
│── requirements.txt
│── Dockerfile
│── .env
│── README.md
```

## Future Improvements

- Persistent vector database
- Authentication system
- Streaming responses
- Source citations
- PDF page references
- Redis based memory
- Hybrid search
- Docker deployment

