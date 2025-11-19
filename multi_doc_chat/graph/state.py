from typing import TypedDict, List, Any


class GraphState(TypedDict):
    input: str
    chat_history: List[Any]
    orchestrator: Any
    route: str
    output: Any
