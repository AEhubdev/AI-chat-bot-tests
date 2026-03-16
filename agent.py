import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType


def execute_agent_query(df, query, api_key):
    """
    Creates an agent that can modify the dataframe and returns the result.
    """
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=api_key
    )

    # We include 'df' in the local scope and instruct the agent to modify it
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True,
        # Better instructions for state persistence
        prefix="You are an expert data scientist. When asked to modify data (delete, merge, pivot), "
               "ensure the final line of your code returns the modified dataframe 'df'."
    )

    try:
        # We use 'with_tool_outputs' logic effectively here
        response = agent.invoke({"input": query})
        return response["output"]
    except Exception as e:
        return f"Error: {str(e)}"