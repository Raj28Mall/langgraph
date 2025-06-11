import os
import subprocess
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

@tool
def run_terminal_command(command: str) -> str:
    """Executes a command in the terminal and returns its output and errors.

    Args:
        command (str): The command to execute in the terminal.

    Returns:
        str: A string containing the exit code, standard output, and standard error.
    """
    if not command:
        return "Error: No command provided."
    try:
        result = subprocess.run(
            command,
            shell=True,  #risky
            capture_output=True,
            text=True,
            timeout=30,
            check=False # To not raise an exception on non-zero exit codes
        )
        
        output = f"Exit Code: {result.returncode}\n"
        if result.stdout:
            output += f"--- STDOUT ---\n{result.stdout}\n"
        if result.stderr:
            output += f"--- STDERR ---\n{result.stderr}\n"
            
        return output.strip()
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds."
    except Exception as e:
        return f"An error occurred while trying to run the command {command}: {str(e)}"

tools = [run_terminal_command]
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0)

SYSTEM_PROMPT = """
You are a helpful AI assistant with the ability to execute commands in a user's terminal.

Workflow:
1.  Analyze: Understand the user's request.
2.  Plan: Break down the request into a series of terminal commands. Think step-by-step.
3.  Execute: Use the `run_terminal_command` tool to execute one command at a time.
4.  Observe: Look at the output from the tool (STDOUT, STDERR, Exit Code).
5.  Reason:
    - If the command was successful (Exit Code 0) and achieved a step, plan the next command.
    - If the command failed (non-zero Exit Code), analyze the error in STDERR and try to fix your command.
    - If the user's request is complete, report the final result to the user.

Important Rules:
- DO NOT AGREE TO REMOVE OR DELETE ANY FILES OR FOLDERS. If asked to do so, you must politely decline.
- Only use the `run_terminal_command` tool.
- List files (`ls -l` on Linux/macOS or `dir` on Windows) to see the current state of the directory when you need to understand your environment.
- When you have successfully completed the entire request, summarize what you did and present the final result to the user.
"""

agent = create_react_agent(llm, tools)

def run_agent():
    """
    Starts an interactive command-line session with the agent.
    """
    print("🤖 Terminal Agent (LangGraph ReAct) is ready. Use with caution.")
    print("   Type 'exit' or 'quit' to end.")
    
    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("🤖 AI: Goodbye!")
            break
        
        if not user_input:
            continue
            
        inputs = {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_input)]}
        
        try:
            print("🤖 AI thinking...", end="", flush=True)
            result = agent.invoke(inputs)
            final_answer = result['messages'][-1].content
            
            print("\r" + " " * 15 + "\r", end="") 
            print(f"🤖 AI:\n{final_answer}")

        except Exception as e:
            print("\r" + " " * 15 + "\r", end="") 
            print(f"An error occurred while running the agent: {e}")

if __name__ == "__main__":
    run_agent()
