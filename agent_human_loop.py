from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, Tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.utilities import SerpAPIWrapper
import os

load_dotenv()

openai_api_key = os.getenv('OPENAI_KEY')
search_api_key = os.getenv('SERPAPI_KEY')

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_feedback: str
    feedback_needed: bool

# Annotated allows you to attach metadata to a type hint, providing additional context about the function
# Sequence represents any ordered collection of items not just lists, tuples or str, allowing for more flexibility in the types of collections that can be used.
    # handles the state updates for sequences such as adding a new message to the chat history
# BaseMessage 
# HumanMessage
# AIMessage
# ToolMessage
# SystemMessage
# add_Messages is a reducer function - is a rule that tells us how to merge new data into the current state (we dont want to replace the state value entirely just append)

@tool
def search_web(query: str) -> str:
    """Search the web for information using SerpAPI."""
    search = SerpAPIWrapper(serpapi_api_key=search_api_key)
    search_results = search.run(query)
    return search_results

tools = [search_web]

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, openai_api_key=openai_api_key).bind_tools(tools)

def agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="""You are a helpful AI assistant. You can perform web searches using the 'search_web' tool."""
    )
    messages = state.get("messages", [])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [system_prompt] + messages
    response = llm.invoke(messages)
    
    # Set feedback_needed flag after generating a response
    feedback_needed = True
    if any(isinstance(msg, HumanMessage) and "feedback" in msg.content.lower() for msg in messages[1:]):
        feedback_needed = False
    
    return {
        "messages": [response],
        "feedback_needed": feedback_needed
    }

def human_feedback(state: AgentState) -> AgentState:
    """Collects feedback from the user on the agent's response."""
    feedback = state.get("user_feedback", "")
    if feedback:
        feedback_message = HumanMessage(content=feedback)
        return {"messages": [feedback_message]}
    return state

def should_continue(state: AgentState) -> str:
    """Determine the next action based on the current state."""
    messages = state.get("messages", [])
    if not messages:
        return "agent"
    
    last_message = messages[-1]
    
    # If the last message has tool calls, execute tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # If we need feedback and don't have processed feedback yet
    if state.get("feedback_needed", False) and not state.get("user_feedback"):
        return "human_feedback"
    
    # If we have unprocessed feedback, process it
    if state.get("user_feedback"):
        return "human_feedback"
    
    # Otherwise, continue with agent or end
    return END

builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_node("human_feedback", human_feedback)
builder.add_node("tools", ToolNode(tools))
builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")
builder.add_edge("human_feedback", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=['human_feedback'])

def run_agent():
    initial_input = {
        "messages": [HumanMessage(content="What is the latest news on AI?")]
    }

    thread = {"configurable": {"thread_id": "1"}}
    print("Starting the agent...")

    for event in graph.stream(initial_input, thread):
        if "agent" in event:
            print(f"Agent response: {event['agent']['messages'][-1].content}")
        elif "tools" in event:
            print(f"Tool call: {event}")
        
    state = graph.get_state(thread)
    if state.next:
        print(f"Interrupting the agent for user feedback...")
        print("current State")
        if state.values.get("messages"):
            print(f"Last message: {state.values['messages'][-1].content}")

        user_fedback = input("Please provide your feedback on the agent's response: ")
        graph.update_state(thread, {"user_feedback": user_fedback})
        for event in graph.stream(None, thread):
            if "agent" in event:
                print(f"Agent response after feedback: {event['agent']['messages'][-1].content}")
            elif "tools" in event:
                print(f"Tool call after feedback: {event}")
    
if __name__ == "__main__":
    run_agent()
    print("Agent execution completed.")


    