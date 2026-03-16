import streamlit as st
import pandas as pd
from agent import create_excel_agent
import io

st.set_page_config(page_title="AI Excel Assistant", layout="wide")

st.title("📊 AI Excel Agent")
st.subheader("Upload, Transform, and Chat with your Data")

# Sidebar for API Key
with st.sidebar:
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    st.info("This agent uses GPT-4o to manipulate your spreadsheet.")

# 1. File Upload
uploaded_file = st.file_uploader("Upload your Excel/CSV file", type=['xlsx', 'csv'])

if uploaded_file and api_key:
    # Load data into session state to keep changes between chat turns
    if "df" not in st.session_state:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)

    # Preview
    st.write("### Data Preview")
    st.dataframe(st.session_state.df.head(10))

    # 2. Chat Interface
    st.write("---")
    st.write("### Chat with Agent")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ex: 'Delete the first column' or 'Create a pivot table of sales by region'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Initialize Agent
            agent = create_excel_agent(st.session_state.df, api_key)

            # The agent executes code.
            # Note: For complex transformations like "delete column",
            # instructions should tell the agent to modify the df.
            response = agent.run(prompt)
            st.markdown(response)

            # Save response to history
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Rerender to show changes if any occurred in the memory (simulated)
            # In a production app, you would capture the agent's code to update st.session_state.df

    # 3. Export Functionality
    st.write("---")
    st.write("### Export Results")

    # Export to Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        st.session_state.df.to_excel(writer, index=False, sheet_name='Sheet1')

    st.download_button(
        label="Download Updated Excel",
        data=buffer.getvalue(),
        file_name="ai_processed_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

elif not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to start.")