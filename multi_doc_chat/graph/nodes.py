from multi_doc_chat.graph.orchestrator import Orchestrator
from multi_doc_chat.logger import GLOBAL_LOGGER as log

"""
Each node is a simple function that returns a dict: {"output": <Message or dict>}
Graph wiring is done in graph_builder.
"""


# Appends the current step into existing steps in the state
def _append_step(state, step):
    steps = state.get("steps", [])
    return steps + [step]


def router_node(state):
    """
    LLM router Decides which agent to route to:
    - If the query is document related → RAG agent
    - Else if it requires tools → tool agent
    - Otherwise → reasoning agent
    """
    orchestrator: Orchestrator = state["orchestrator"]
    user_query = state["input"]
    chat_history = state.get("chat_history", [])

    log.info("Router node evaluating query")

    # Calls {route_query} function in orchestrator which decides which node to be routed to:
    routing_decision = orchestrator.route_query(user_query, chat_history)

    log.info("Router node decision", route=routing_decision)

    return {"route": routing_decision, "steps": _append_step(state, "router")}


def rag_node(state):
    orchestrator: Orchestrator = state["orchestrator"]
    user_query = state["input"]
    chat_history = state.get("chat_history", [])

    log.info("RAG node invoked")

    # Calls {run_rag} function in orchestrator which runs the RAG Pipeline:
    response = orchestrator.run_rag(user_query, chat_history)

    return {
        "output": response,
        "steps": _append_step(state, "rag"),
    }


def reasoning_node(state):
    orchestrator: Orchestrator = state["orchestrator"]
    query = state["input"]

    log.info("Reasoning node invoked")

    # Calls {run_reasoning} function in orchestrator which runs the Reasoning Pipeline:
    response = orchestrator.run_reasoning(query)

    return {
        "output": response,
        "steps": _append_step(state, "reasoning"),
    }


def tool_node(state):
    orchestrator: Orchestrator = state["orchestrator"]
    query = state["input"]

    log.info("Tools node invoked")

    # Calls {run_tools} function in orchestrator which runs the Reasoning Pipeline:
    response = orchestrator.run_tools(query)

    return {
        "output": response,
        "steps": _append_step(state, "tools"),
    }
