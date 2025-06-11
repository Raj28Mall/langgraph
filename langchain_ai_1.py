import os
import random
from typing import Dict, TypedDict, Union, Optional, Any, List
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

def process(state: AgentState) -> AgentState:
    response= llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print(f"AI: {response.content}")
    return state

graph = StateGraph(AgentState)

graph.set_entry_point('process')
graph.add_node('process', process)
graph.set_finish_point('process')
agent=graph.compile()

conversation_history=[]

user_input= input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result= agent.invoke({"messages":conversation_history})
    conversation_history= result['messages']
    user_input=input("Enter: ")
