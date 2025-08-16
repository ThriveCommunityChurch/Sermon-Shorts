import operator
from typing import Annotated, List, Tuple, TypedDict
from langgraph.graph import MessagesState

class AgentState(TypedDict):
    messages: MessagesState
    filePath: Annotated[str, "The file path of the file we're transcribing."]