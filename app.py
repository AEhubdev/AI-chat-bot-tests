import streamlit as st
import os
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIG ---
st.set_page_config(page_title="Pro NotebookLM Local", layout="wide")
# Using the 8b model for better reasoning
MODEL_NAME = "llama3.1:8b"
EMBED_MODEL = "nomic-embed-text"

# State Initialization
for key in ["messages", "files", "vector_db"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key != "vector_db" else None

# --- SIDEBAR: SOURCE PROCESSING ---
with st.sidebar:
    st.header("📂 Research Library")
    uploaded_files = st.file_uploader("Upload PDFs (Deka Reports, BIBs, etc.)", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.files]
        if new_files:
            with st.status("Deep Document Analysis...") as status:
                all_chunks = []
                # UPGRADE: Larger chunks to capture full financial tables and headers
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=300,
                    separators=["\n\n", "\n", ".", " "]
                )

                for f in new_files:
                    path = f"temp_{uuid.uuid4().hex}.pdf"
                    with open(path, "wb") as b:
                        b.write(f.getvalue())
                    try:
                        loader = PyPDFLoader(path)
                        docs = loader.load()
                        for i, doc in enumerate(docs):
                            doc.metadata["source"] = f.name
                            doc.metadata["page"] = i + 1
                        all_chunks.extend(splitter.split_documents(docs))
                        st.session_state.files.append(f.name)
                    finally:
                        if os.path.exists(path): os.remove(path)

                emb = OllamaEmbeddings(model=EMBED_MODEL)
                if st.session_state.vector_db is None:
                    st.session_state.vector_db = Chroma.from_documents(all_chunks, emb)
                else:
                    st.session_state.vector_db.add_documents(all_chunks)
                status.update(label="Library Synced", state="complete")

    if st.button("Purge Library"):
        st.session_state.clear()
        st.rerun()


# --- ADVANCED LOGIC: MULTI-QUERY EXPANSION ---
def get_expanded_context(query):
    # Use the 8b model to think of better search terms
    llm = ChatOllama(model=MODEL_NAME, temperature=0)

    variation_prompt = ChatPromptTemplate.from_template(
        "You are an expert search assistant. Given the user question, generate 3 specific search queries "
        "to find the relevant financial data in PDF documents. \nQuestion: {query}"
    )
    search_chain = variation_prompt | llm | StrOutputParser()
    variations = search_chain.invoke({"query": query}).split("\n")

    # Increase 'k' to get more context for the 8b model to chew on
    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 6})
    all_retrieved_docs = []
    for q in variations + [query]:
        all_retrieved_docs.extend(retriever.invoke(q))

    # Deduplicate
    unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}.values()
    return list(unique_docs)


# --- MAIN CHAT ---
st.title("📝 Pro Research Assistant (8B)")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ex: 'Vergleiche die Kosten und das Risiko beider Dokumente'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if st.session_state.vector_db:
        with st.chat_message("assistant"):
            with st.spinner("Synthesizing Cross-Document Intelligence..."):
                docs = get_expanded_context(prompt)

                context_str = "\n\n".join([
                    f"--- SOURCE: {d.metadata['source']} (PAGE {d.metadata.get('page', '?')}) ---\n{d.page_content}"
                    for d in docs
                ])

                # UPGRADE: Professional prompt tailored for German financial docs
                system_prompt = f"""
                You are a Senior Financial Investment Analyst. Your task is to analyze and compare regulatory documents.

                DOCUMENTS IDENTIFIED IN CONTEXT:
                - Basisinformationsblatt (BIB/KID): Focuses on Risk Class (1-7), Future Scenarios, and Cost Disclosure.
                - Halbjahresbericht (HJB) / Jahresbericht: Focuses on actual holdings, past performance, and market reports.

                YOUR TASK:
                1. Explicitly identify the document type for each source.
                2. If the user asks for a comparison or overview, structure your answer by:
                   - Purpose & Regulatory Function
                   - Risk & Performance (Projections vs. Reality)
                   - Cost Structures
                   - Portfolio Details
                3. Use a Markdown Table for any data comparisons.
                4. ALWAYS cite the source filename for every claim.

                Language: If the user asks in German, respond in German. Otherwise, English.

                CONTEXT:
                {context_str}
                """

                llm = ChatOllama(model=MODEL_NAME, temperature=0)
                final_response = llm.invoke([
                    ("system", system_prompt),
                    ("human", prompt)
                ])

                res_content = final_response.content
                st.markdown(res_content)

                # Show citations in a clean way
                with st.expander("View Retrieved Context Segments"):
                    for d in docs:
                        st.caption(f"From: {d.metadata['source']}")
                        st.text(d.page_content[:300] + "...")

                st.session_state.messages.append({"role": "assistant", "content": res_content})
    else:
        st.warning("Please upload your PDFs to the library first.")