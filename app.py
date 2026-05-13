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

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Streamlit page configuration
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

    groq_api_key = st.text_input(
        "Enter Groq API Key",
        type="password"
    )

    session_id = st.text_input(
        "Session ID",
        value="default_session"
    )

# Session state initialization
if "store" not in st.session_state:
    st.session_state.store = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Main application
if groq_api_key:

    # Initialize LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
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

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.read())
                    temp_pdf_path = temp_file.name

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

        # Contextualization system prompt
        contextualize_q_system_prompt = """
        Given the chat history and latest user question,
        formulate a standalone question.

        Do not answer the question.
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
        You are a helpful assistant.

        Use the retrieved context to answer the question.

        If the answer is not available,
        say you do not know.

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

        # Question answering chain
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt
        )

        # Retrieval chain
        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )

        # Session history function
        def get_session_history(
            session: str
        ) -> BaseChatMessageHistory:

            if session not in st.session_state.store:

                st.session_state.store[session] = ChatMessageHistory()

            return st.session_state.store[session]

        # Conversational RAG chain
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

        # Display previous chat messages
        for message in st.session_state.messages:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

        # Process user query
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

                with st.spinner("Thinking..."):

                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={
                            "configurable": {
                                "session_id": session_id
                            }
                        }
                    )

                    answer = response["answer"]

                    st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

# Warning message
else:

    st.warning(
        "Please enter your Groq API Key."
    )