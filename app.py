import streamlit as st
import pandas as pd
import io
import os

# Updated Imports for 2026 LangChain
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# UI Config
st.set_page_config(page_title="AI Data Architect Pro", layout="wide", page_icon="🚀")


# --- Agent Logic (Embedded directly to prevent ModuleNotFoundError) ---
def run_ai_transformation(df, user_query, api_key):
    try:
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=api_key
        )

        # We use agent_type="openai-tools" which is the 2026 standard
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            agent_type="openai-tools",  # More stable than legacy agent_types enum
            allow_dangerous_code=True,
            prefix="""You are a world-class Data Engineer. 
            When asked to 'delete', 'drop', 'rename', or 'modify' columns/rows:
            1. Perform the operation on the dataframe 'df'.
            2. Explain exactly what you did.
            3. Always mention the resulting number of columns or rows."""
        )

        response = agent.invoke({"input": user_query})
        return response["output"]
    except Exception as e:
        return f"⚠️ Analysis Error: {str(e)}"


# --- Main Streamlit App ---
st.title("🚀 AI Data Architect Pro")
st.markdown("##### Transform Excel & CSV with natural language.")

# Sidebar API Management
with st.sidebar:
    st.header("🔑 Authentication")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if not api_key:
        st.warning("Enter your API key to enable the AI.")

    st.divider()
    st.header("🧹 Session Controls")
    if st.button("Reset Application"):
        st.session_state.clear()
        st.rerun()

# File Upload
uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])

if uploaded_file:
    # Initialize Data
    if "df" not in st.session_state:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)

    # UI Columns
    preview_col, chat_col = st.columns([1.5, 1])

    with preview_col:
        st.subheader("📊 Live Data Preview")
        st.dataframe(st.session_state.df, use_container_width=True, height=450)

        # Download Handler
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            st.session_state.df.to_excel(writer, index=False)

        st.download_button(
            label="💾 Download Current File",
            data=buffer.getvalue(),
            file_name="transformed_data.xlsx",
            mime="application/vnd.ms-excel"
        )

    with chat_col:
        st.subheader("💬 AI Agent")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Chat display
        container = st.container(height=350)
        for m in st.session_state.messages:
            container.chat_message(m["role"]).write(m["content"])

        # Query Input
        if prompt := st.chat_input("Ex: 'Delete the Unnamed columns'"):
            if not api_key:
                st.error("Missing API Key!")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                container.chat_message("user").write(prompt)

                with st.status("Engineer is working on your data..."):
                    result = run_ai_transformation(st.session_state.df, prompt, api_key)

                st.session_state.messages.append({"role": "assistant", "content": result})
                container.chat_message("assistant").write(result)

                # To ensure persistent changes, if the agent says it modified something,
                # we force a rerun to refresh the 'st.dataframe' view.
                st.rerun()
else:
    st.info("Waiting for a CSV or Excel file to be uploaded.")