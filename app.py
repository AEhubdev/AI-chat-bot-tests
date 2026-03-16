import streamlit as st
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. SETUP MODEL (Switch to the Coder model)
MODEL_NAME = "qwen2.5-coder:3b"

st.set_page_config(page_title="Pro Data Agent", layout="wide")
st.title("⚡ Ultra-Fast Local Data Agent")

# Sidebar for status
with st.sidebar:
    st.info(f"Model: {MODEL_NAME}")
    if st.button("Clear History & Reset"):
        st.session_state.clear()
        st.rerun()

uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

if uploaded_file:
    if "df" not in st.session_state:
        st.session_state.df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(
            uploaded_file)

    st.dataframe(st.session_state.df, height=300)

    if prompt := st.chat_input("What should I do with the data?"):
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Executing..."):
                # We use a lower temperature (0) for zero 'creativity' and more 'logic'
                llm = ChatOllama(model=MODEL_NAME, temperature=0)

                # We force the agent to use the specific Python tool
                agent = create_pandas_dataframe_agent(
                    llm,
                    st.session_state.df,
                    verbose=True,
                    allow_dangerous_code=True,
                    handle_parsing_errors=True,
                    max_iterations=5  # Stops it from looping forever
                )

                try:
                    # Explicitly tell it to use the tool in the prompt suffix
                    response = agent.invoke({
                        "input": f"{prompt}. IMPORTANT: Use the 'python_repl_ast' tool to perform this. Do not explain, just execute."
                    })
                    st.write(response["output"])
                    st.rerun()  # Immediately show the new data state
                except Exception as e:
                    st.error(f"Execution took too long or failed: {e}")