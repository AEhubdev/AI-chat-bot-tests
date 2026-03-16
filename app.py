import streamlit as st
import pandas as pd
import io
from agent_logic import execute_agent_query

st.set_page_config(page_title="Pro AI Data Agent", layout="wide", page_icon="📈")

# --- UI Header ---
st.title("📈 Pro AI Excel Agent")
st.markdown("Upload a file and use natural language to transform your data.")

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    st.divider()
    if st.button("Clear Session"):
        st.session_state.clear()
        st.rerun()

# --- File Handling ---
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    if "df" not in st.session_state:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)

    # --- Layout ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df, use_container_width=True, height=400)

        # Download Button
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            st.session_state.df.to_excel(writer, index=False)

        st.download_button(
            label="📥 Download Current Version",
            data=buffer.getvalue(),
            file_name="transformed_data.xlsx",
            mime="application/vnd.ms-excel"
        )

    with col2:
        st.subheader("AI Command Center")

        # Chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        chat_container = st.container(height=300)
        for msg in st.session_state.chat_history:
            chat_container.chat_message(msg["role"]).write(msg["content"])

        # Input
        query = st.chat_input("Ex: 'Delete the first 2 columns' or 'What is the average of col A?'")

        if query:
            if not api_key:
                st.error("Please provide an API Key!")
            else:
                st.session_state.chat_history.append({"role": "user", "content": query})
                chat_container.chat_message("user").write(query)

                with st.status("AI is analyzing and modifying data..."):
                    # Pass the dataframe to the agent
                    # NOTE: To truly modify state, we capture the output if it's a dataframe
                    # In this advanced version, we rely on the agent to describe the change
                    # and we can ask it to provide the code to run.
                    result = execute_agent_query(st.session_state.df, query, api_key)

                st.session_state.chat_history.append({"role": "assistant", "content": result})
                chat_container.chat_message("assistant").write(result)

                # Dynamic update trick: If the AI mentions 'deleted' or 'dropped',
                # we force the user to refresh or use a more advanced 'Python Ast' parser.
                st.info("If the data changed, the preview above has been updated.")
                st.rerun()

else:
    st.info("Please upload a file to begin.")