import os
import subprocess
from typing import TypedDict, Annotated, Sequence
from langchain.tools import tool
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


@tool
def run_shell_command(command: str) -> str:
    """
    Executes a shell command on the user's computer and returns the output.
    
    Args:
        command: The shell command to execute (e.g., 'ls -l', 'mkdir new_dir').
                 Can be used for directory creation, navigation (by chaining commands like 'cd dir && ls'),
                 file creation ('echo "hello" > file.txt'), and reading files ('cat file.txt').
    Returns:
        A string containing the standard output and standard error from the command.
    """
    print(f"--- Executing Shell Command: {command} ---")
    try:
        # Using shell=True allows for complex commands, but carries security risks.
        # See the security warning in the documentation.
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            cwd=os.getcwd() # Ensures command runs in the current directory of the script
        )
        if result.returncode == 0:
            return f"Output:\n{result.stdout}"
        else:
            return f"Error (Exit Code {result.returncode}):\n{result.stderr}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

@tool
def get_running_processes() -> str:
    """
    Reads the current processes running on the computer.
    Uses 'ps aux' on Unix-like systems and 'tasklist' on Windows.
    Returns a string containing the list of processes.
    """
    print("--- Getting Running Processes ---")
    command = "ps aux"
    return run_shell_command.invoke(command)

@tool
def get_ram_usage() -> str:
    """
    Reads the current RAM usage of the computer.
    Uses 'free -h' on Linux and 'vm_stat' on macOS.
    Provides basic memory info for Windows.
    Returns a string containing the memory usage details.
    """
    print("--- Getting RAM Usage ---")
    command = "free -h"
    return run_shell_command.invoke(command)

class AgentState(TypedDict):
    """
    The state of our agent. It's a dictionary that tracks messages.
    'add_messages' means that new messages are always added to the existing list.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]

tools = [run_shell_command, get_running_processes, get_ram_usage]


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0).bind_tools(tools)

def should_continue(state: AgentState) -> str:
    """
    Determines the next step for the agent.
    If the last message has tool calls, we execute the tools.
    Otherwise, we end the process.
    """
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "continue"
    return "end"

def call_model(state: AgentState):
    """
    The first node in our graph. It calls the LLM to decide what to do.
    """
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

graph = StateGraph(AgentState)
tool_node = ToolNode(tools)
graph.set_entry_point("agent") 
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools", 
        "end": END     
    }
)
graph.add_edge('tools', 'agent')
agent = graph.compile()

def run_agent():
    print("--- LangGraph ReAct Terminal Agent ---")
    print("Type 'exit' or 'quit' to end the session.")
    while True:
        try:
            user_input = input("\nUser > ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            inputs = {"messages": [HumanMessage(content=user_input)]}
            
            final_state = None
            for output in agent.stream(inputs, {"recursion_limit": 10}):
                for key, value in output.items():
                    print(f"Output from node '{key}':")
                    print("---")
                final_state = output

            if final_state and 'agent' in final_state and final_state['agent']['messages']:
                final_response = final_state['agent']['messages'][-1].content
                print("\nAgent >")
                print(final_response)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_agent()
        
