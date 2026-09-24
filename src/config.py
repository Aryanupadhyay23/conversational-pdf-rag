import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")

# LangSmith Tracing configurations
LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "conversational-pdf-rag")

os.environ["HF_TOKEN"] = HF_TOKEN
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if LANGCHAIN_TRACING_V2 and LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# Model configurations
DEFAULT_LLM_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIMENSION = 768

# Self-RAG configurations (Optimized chunk size for high-precision retrieval)
MAX_RETRIES = 2
RETRIEVER_K = 5
CHUNK_SIZE = 1000       # ~200-250 tokens: high semantic density for 768-dim embeddings
CHUNK_OVERLAP = 200     # 20% overlap: preserves cross-chunk contextual continuity

# Hybrid Retrieval configurations (EnsembleRetriever weights)
BM25_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.5

def get_llm(temperature: float = 0.3) -> BaseChatModel:
    """Return standard generation chat model initialized via LangChain's native init_chat_model."""
    return init_chat_model(
        DEFAULT_LLM_MODEL,
        model_provider="groq",
        api_key=GROQ_API_KEY,
        temperature=temperature
    )

def get_eval_llm(temperature: float = 0.0) -> BaseChatModel:
    """Return deterministic grading model initialized via LangChain's native init_chat_model."""
    return init_chat_model(
        DEFAULT_LLM_MODEL,
        model_provider="groq",
        api_key=GROQ_API_KEY,
        temperature=temperature
    )

def get_embeddings():
    """Return LangChain native Google Gemini embeddings model with 768 output dimensions."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    api_key = GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=api_key,
        output_dimensionality=EMBEDDING_DIMENSION
    )
