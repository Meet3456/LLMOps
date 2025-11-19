from langgraph.types import Message
from multi_doc_chat.graph.orchestrator import RAGAgent, ReasoningAgent, ToolAgent
from multi_doc_chat.logger import GLOBAL_LOGGER as log

"""
Each node is a simple function that returns a dict: {"output": <Message or dict>}
Graph wiring is done in graph_builder.
"""

# ========================= ROUTER NODE ===========================
def router_node(state):
    """
    Decides which agent to route to:
    - If the query is document related → RAG agent
    - Else if it requires tools → tool agent
    - Otherwise → reasoning agent
    """

    user_query = state["input"]
    orchestrator = state["orchestrator"]

    log.info("Router received query", query=user_query)

    # 1) Check if query requires document context
    if orchestrator.retriever.is_document_query(user_query):
        log.info("Routing → RAG agent")
        return {"route": "rag"}

    # 2) Check if query needs a tool
    if orchestrator.tool_detector.needs_tools(user_query):
        log.info("Routing → Tool agent")
        return {"route": "tools"}

    # 3) Otherwise → reasoning agent
    log.info("Routing → Reasoning agent")
    return {"route": "reasoning"}


# ========================= RAG NODE ==============================
def rag_node(state):
    orchestrator = state["orchestrator"]
    query = state["input"]
    chat_history = state["chat_history"]

    rag = RAGAgent(orchestrator)
    return {"output": rag.run(query, chat_history)}


# ========================= REASONING NODE =========================
def reasoning_node(state):
    orchestrator = state["orchestrator"]
    query = state["input"]

    agent = ReasoningAgent(orchestrator)
    return {"output": agent.run(query)}


# ========================= TOOL NODE ==============================
def tool_node(state):
    orchestrator = state["orchestrator"]
    query = state["input"]

    agent = ToolAgent(orchestrator)
    return {"output": agent.run(query)}
