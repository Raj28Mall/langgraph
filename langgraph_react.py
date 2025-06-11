from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

document_content=""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
    """Returns sum of 2 numbers"""
    return a+b

@tool
def subtract(a: int, b: int):
    """Returns subtraction of 2 numbers"""
    return a-b

@tool
def multiply(a: int, b: int):
    """Returns multiplication of 2 numbers"""
    return a*b


tools=[add, subtract, multiply]

llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7).bind_tools(tools)

def llm_call(state: AgentState) -> AgentState:
    response= llm.invoke(state['messages']) 
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages= state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(AgentState)

graph.set_entry_point("our_agent")
graph.add_node("our_agent", llm_call)
tool_node =ToolNode(tools=tools)
graph.add_node("tools", tool_node)
graph.add_edge("tools", "our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        'continue':'tools',
        'end':END
    }
)
agent=graph.compile()

def print_stream(stream):
    for s in stream:
        message= s['messages'][-1]
        if isinstance(message, tuple):  
            print(message)
        else:
            message.pretty_print()

inputs = {
    "messages": [
        SystemMessage(content="You are my AI assistant, you must fulfill the user's request which may require multiple tool calls."),
        ("user", "Add 40 + 12 and then multiply the result by 6. And then tell me a joke with some premise to it")
    ]
}
print_stream(agent.stream(inputs, stream_mode= "values"))