from langgraph.graph import StateGraph, END
from multi_doc_chat.graph.nodes import (
    router_node,
    rag_node,
    reasoning_node,
    tool_node,
)
from multi_doc_chat.graph.state import GraphState


def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("router")

    # conditional routing
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "rag": "rag",
            "reasoning": "reasoning",
            "tools": "tools",
        },
    )

    graph.add_edge("rag", END)
    graph.add_edge("reasoning", END)
    graph.add_edge("tools", END)

    return graph.compile()
