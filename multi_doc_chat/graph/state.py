from typing import TypedDict, List, Any, Literal


class GraphState(TypedDict):
    input: str
    chat_history: List[Any]
    orchestrator: Any
    route: Literal["rag", "reasoning", "tools"]
    output: Any
    steps: List[str]
