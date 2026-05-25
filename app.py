import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.chat_history import BaseChatMessageHistory

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.chat_message_histories import (
    ChatMessageHistory
)

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.environ.get("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

# Streamlit page config
st.set_page_config(
    page_title="Conversational PDF RAG",
    page_icon="📚",
    layout="wide"
)

# App title
st.title("📚 Conversational PDF RAG")
st.write("Upload PDF files and chat with their content.")

# Sidebar
with st.sidebar:
    st.header("Settings")

    session_id = st.text_input(
        "Session ID",
        value="default_session"
    )

# Session state
if "store" not in st.session_state:
    st.session_state.store = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Run app if API key exists
if groq_api_key:

    # Initialize Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0.3
    )

    # Upload PDFs
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type="pdf",
        accept_multiple_files=True
    )

    # Process PDFs
    if uploaded_files:

        documents = []

        with st.spinner("Processing PDFs..."):

            for uploaded_file in uploaded_files:

                # Save uploaded pdf temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.read())
                    temp_pdf_path = temp_file.name

                # Load pdf
                loader = PyPDFLoader(temp_pdf_path)
                docs = loader.load()

                documents.extend(docs)

        st.success("PDFs processed successfully!")

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500
        )

        splits = text_splitter.split_documents(documents)

        # Create vector database
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever()

        # Query reformulation prompt
        contextualize_q_system_prompt = """
        You are an expert query reformulator.

        Your job is to analyze the chat history and latest user question
        to create a standalone search query.

        Rules:
        1. Do not answer the question.
        2. Do not add explanations.
        3. Output only the standalone question.
        4. If no history is needed, return the question as it is.
        """

        # Contextualization prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # History aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm,
            retriever,
            contextualize_q_prompt
        )

        # QA system prompt
        system_prompt = """
        You are an expert assistant for document question answering.

        Answer the question only from the provided context.

        Rules:
        1. Do not hallucinate.
        2. If information is unavailable, clearly say so.
        3. Keep answers clear and structured.
        4. Use markdown formatting when useful.

        Context:
        {context}
        """

        # QA prompt
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        # QA chain
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )

        # Retrieval chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        # Session history
        def get_session_history(
            session: str
        ) -> BaseChatMessageHistory:

            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()

            return st.session_state.store[session]

        # Conversational chain
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        st.divider()

        # User input
        user_input = st.chat_input(
            "Ask a question about your PDFs..."
        )

        # Display previous messages
        for message in st.session_state.messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Process query
        if user_input:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_input
                }
            )

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):

                # Stream response tokens
                def response_generator():

                    for chunk in conversational_rag_chain.stream(
                        {"input": user_input},
                        config={
                            "configurable": {
                                "session_id": session_id
                            }
                        }
                    ):

                        if "answer" in chunk:
                            yield chunk["answer"]

                answer = st.write_stream(
                    response_generator
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

# Show error if key missing
else:

    st.error(
        "Missing API Key Configuration. Please add GROQ_API_KEY in environment variables."
    )