import os
from rich.console import Console
from rich.markdown import Markdown

from multi_doc_chat.graph.builder import build_graph
from multi_doc_chat.graph.orchestrator import Orchestrator

console = Console()

# ===========================================================
# CONFIGURATION — CHANGE THIS TO YOUR SESSION INDEX PATH
# ===========================================================
INDEX_PATH = "faiss_index/session_18_nov_2025_3:54_pm_5d9f"

# ===========================================================
# INITIALIZE ORCHESTRATOR + GRAPH
# ===========================================================
console.print("[bold cyan]Initializing orchestrator and graph...[/bold cyan]")

orch = Orchestrator(index_path=INDEX_PATH)
graph = build_graph()

# Chat history for chatbot mode
chat_history = []

console.print("[green]Initialization complete! Chatbot ready.[/green]\n")

# ===========================================================
# CHAT LOOP
# ===========================================================
while True:
    user_input = console.input("[bold magenta]You:[/bold magenta] ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        console.print("[yellow]Exiting chat. Goodbye![/yellow]")
        break

    # Build state for graph
    state = {
        "input": user_input,
        "chat_history": chat_history,
        "orchestrator": orch,
        "route": None,
        "output": None,
    }

    # Run through LangGraph
    result_state = graph.invoke(state)
    output = result_state.get("output", {})

    # Extract data
    otype = output.get("type", "unknown")
    content = output.get("content").encode("utf-8", "ignore").decode()
    reasoning = output.get("reasoning")
    tools_used = output.get("executed_tools")

    # Print output
    console.print(f"\n[bold cyan]Route Selected:[/bold cyan] {otype.upper()}")
    console.print("\n[bold green]Assistant:[/bold green]")
    console.print(Markdown(content if content else "`<no content>`"))

    # Optional: show reasoning
    # if reasoning:
    #     console.print("\n[bold yellow]Reasoning:[/bold yellow]")
    #     console.print(Markdown(str(reasoning)))

    # Optional: show tool results
    if tools_used:
        console.print("\n[bold magenta]Executed Tools:[/bold magenta]")
        console.print(Markdown(str(tools_used)))

    # Store in chat history
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": content})

    console.print("\n" + "-" * 60 + "\n")
