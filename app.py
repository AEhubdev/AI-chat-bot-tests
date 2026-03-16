import streamlit as st
import pandas as pd
import io
import os

# Local AI Imports
from langchain_ollama import ChatOllama
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- UI Setup ---
st.set_page_config(page_title="Local Data AI", layout="wide", page_icon="🤖")

st.title("🤖 Local AI Data Agent")
st.markdown("""
    **Privacy First:** This app runs 100% on your local machine using **Ollama**. 
    No data is sent to OpenAI or the cloud.
""")

# --- Model Configuration ---
# You can change 'llama3' to whatever model you 'pulled' in Ollama
LOCAL_MODEL = "llama3"


def get_agent(df):
    llm = ChatOllama(
        model=LOCAL_MODEL,
        temperature=0,
        base_url="http://localhost:11434"  # Default Ollama Port
    )

    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        allow_dangerous_code=True,  # Required for the AI to run Python code on your data
        handle_parsing_errors=True
    )


# --- File Upload ---
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file:
    # Load data into session state
    if "df" not in st.session_state:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)

    # Layout: Preview and Chat
    view_col, chat_col = st.columns([1.5, 1])

    with view_col:
        st.subheader("📊 Data Preview")
        st.dataframe(st.session_state.df, use_container_width=True, height=400)

        # Download Logic
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            st.session_state.df.to_excel(writer, index=False)

        st.download_button(
            label="📥 Download Updated File",
            data=buffer.getvalue(),
            file_name="local_ai_processed.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with chat_col:
        st.subheader("💬 Local Chat")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display history
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        # User Input
        if prompt := st.chat_input("Ex: 'Delete the first column' or 'What is the sum of column X?'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("AI is analyzing local data..."):
                    try:
                        agent = get_agent(st.session_state.df)
                        # We use .invoke for the newer LangChain standard
                        response = agent.invoke({"input": prompt})
                        answer = response["output"]

                        st.write(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})

                        # In many cases, the agent modifies the 'df' object in its own context.
                        # To reflect changes in the UI, we may need to prompt the agent
                        # to specifically confirm it updated the variable.
                        if "deleted" in answer.lower() or "dropped" in answer.lower() or "modified" in answer.lower():
                            st.info("Data modification detected. Refreshing view...")
                            st.rerun()

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.info("Check if Ollama is running (`ollama serve`) in your terminal.")

else:
    st.info("Please upload a file to start.")
    st.image("https://ollama.com/public/ollama.png", width=100)