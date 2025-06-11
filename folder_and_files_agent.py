import os
import subprocess
from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def create_directory(directory_name: str) -> str:
    """Creates a directory with the given name.
    
    Args:
        directory_name (str): Name of the directory to create.

    Returns:
        str: A message indicating success or failure.
    """
    if not directory_name or not isinstance(directory_name, str):
        return "Error: Invalid directory name provided."
    try:
        os.mkdir(directory_name)
        return f"Directory '{directory_name}' was created successfully."
    except FileExistsError:
        return f"Error: Directory '{directory_name}' already exists."
    except Exception as e:
        return f"Error: Failed to create directory '{directory_name}': {str(e)}"

@tool
def create_file(filename: str) -> str:
    """Creates an empty file with the given name.

    Args:
        filename (str): Name for the file to create.

    Returns:
        str: A message indicating success or failure.
    """
    if not filename or not isinstance(filename, str):
        return "Error: Invalid filename provided."
    try:
        with open(filename, 'w') as f:
            f.write("")  
        return f"File '{filename}' was created successfully."
    except Exception as e:
        return f"Failed to create file '{filename}': {str(e)}"

tools = [create_directory, create_file]
SYSTEM_PROMPT = """You are FileAgent, a helpful assistant specializing in linux file system operations. You will help the user create files and directories.

- To create a directory, you must use the 'create_directory' tool.
- To create a file, you must use the 'create_file' tool.
- When a tool is used, you will be given the result. You should then inform the user of what happened.
- If the user asks you to do something you cannot do, politely decline.
- If the user wants to exit, just say goodbye and end the conversation."""

llm = ChatAnthropic(model='claude-3-sonnet-20240229', temperature=0).bind_tools(tools)

def agent_node(state: AgentState) -> AgentState:
    """
    The primary node that invokes the LLM to decide the next action.
    """
    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    
    return {"messages": [response]}

def router(state: AgentState) -> str:
    """
    Determines the next step based on the agent's last message.

    Returns:
        str: 'tools' if the agent decided to use a tool, or '__end__' to finish the turn.
    """
    last_message = state['messages'][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"

graph = StateGraph(AgentState)

graph.set_entry_point('agent')

graph.add_node('agent', agent_node)
graph.add_node('tools', ToolNode(tools=tools))

graph.add_conditional_edges(
    'agent',
    router,
    {
        "tools": "tools",
        "__end__": END
    }
)

graph.add_edge('tools', 'agent')
agent = graph.compile()


def run_agent():
    """
    Runs the FileAgent in a loop, taking user input and printing agent responses.
    """
    print("🤖 FileAgent is ready. Type 'exit' or 'quit' to end the session.")
    
    messages = []
    
    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("🤖 AI: Goodbye!")
            break
        
        human_message = HumanMessage(content=user_input)
        messages.append(human_message)
        
        state = {"messages": messages}
        final_state = agent.invoke(state)
        
        ai_response_message = final_state['messages'][-1]
        messages = final_state['messages']
        
        print(f"🤖 AI: {ai_response_message.content}")

if __name__ == "__main__":
    run_agent()