import streamlit as st
import os
import tempfile
import io

# 2026 Legacy-Safe Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- UI Setup ---
st.set_page_config(page_title="Local NotebookLM", layout="wide")
st.title("📚 Local NotebookLM")

# --- Settings ---
# Pull these in your terminal first:
# ollama pull llama3.2:3b
# ollama pull nomic-embed-text
MODEL_NAME = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("Upload PDF")
    uploaded_pdf = st.file_uploader("Choose a file", type="pdf")

    if uploaded_pdf:
        if "qa_chain" not in st.session_state:
            with st.status("Reading PDF..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.getvalue())
                    tmp_path = tmp.name

                # Load & Split
                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents(docs)

                # Create Vector DB
                embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
                vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)

                # Build Chain
                llm = ChatOllama(model=MODEL_NAME, temperature=0)
                prompt = ChatPromptTemplate.from_template("""
                Answer the question based only on the provided context:
                <context>
                {context}
                </context>
                Question: {input}""")

                document_chain = create_stuff_documents_chain(llm, prompt)
                st.session_state.qa_chain = create_retrieval_chain(vector_db.as_retriever(), document_chain)

                os.remove(tmp_path)  # cleanup
            st.success("Ready!")

# --- Chat Interface ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "qa_chain" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = st.session_state.qa_chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.warning("Please upload a PDF first.")