import argparse
import getpass
import os
from dotenv import load_dotenv
# from IPython.display import Image, display  # Not needed for CLI usage
from langchain_openai import ChatOpenAI, OpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from Classes.agent_state import AgentState
from nodes.transcription_node import transcribe_audio
from nodes.analysis_node import recommend_clips

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"

# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    # Optionally set a default path via SERMON_FILE env var; otherwise tools will auto-detect latest
    # state["filePath"] = os.environ.get("SERMON_FILE")
    response = model.invoke(messages)
    return {"messages": [response]}

# Build tools
tools = [transcribe_audio, recommend_clips]
tool_node = ToolNode(tools)

# Use GPT-4o-mini (gpt-5-mini may not be available yet)
model = ChatOpenAI(model='gpt-4o-mini', api_key=api_key, temperature=0)
model = model.bind_tools(tools)

# Define a new graph
workflow = StateGraph(MessagesState)

# Add Nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Add Edges
workflow.add_edge(START, "agent")

# Conditional edge: agent -> tools or end
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

# After tools, return to agent to allow multiple tool calls (transcribe then analysis)
workflow.add_edge("tools", "agent")

# Set up memory
memory = MemorySaver()

# Finally, we compile it!
app = workflow.compile(checkpointer=memory)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Transcribe sermon and recommend social media clips")
    parser.add_argument("--file", "-f", type=str, help="Path to the sermon audio/video file to transcribe")
    args = parser.parse_args()

    # Set the file path in environment if provided via CLI
    if args.file:
        os.environ["SERMON_FILE_PATH"] = args.file
        print(f"Using provided file: {args.file}")
    else:
        print("No file specified. Will auto-detect latest file in sermon directory.")

    thread = {"configurable": {"thread_id": "3"}}
    inputs = [
        HumanMessage(content=(
            "Please 1) transcribe the sermon audio/video using the transcribe_audio tool, "
            "then 2) analyze the transcription using the recommend_clips tool to output recommended "
            "60-90 second social clip timestamps."
        ))
    ]

    print("Starting sermon transcription and analysis...")
    for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
        event["messages"][-1].pretty_print()

if __name__ == "__main__":
    main()