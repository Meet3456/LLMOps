from typing import Any, List, Literal, Optional, TypedDict


class GraphState(TypedDict):
    input: str
    chat_history: List[Any]
    orchestrator: Any
    docs: Optional[List[Any]]
    route: Literal["rag", "reasoning", "tools"]
    output: Any
    steps: List[str]
