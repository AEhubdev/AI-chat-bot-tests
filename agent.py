import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType


def create_excel_agent(df, api_key):
    """
    Creates an agent that can interact with a pandas dataframe.
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=api_key
    )

    # We use the experimental pandas agent to allow for complex data manipulation
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True  # Required to let the agent run pandas operations
    )
    return agent


def handle_query(agent, query):
    """
    Sends the user query to the agent and returns the updated dataframe or answer.
    """
    # The agent will execute code on the 'df' variable in its memory
    response = agent.run(query)
    return response