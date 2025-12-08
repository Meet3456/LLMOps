from multi_doc_chat.graph.orchestrator import Orchestrator


class OrchestratorManager:
    def __init__(self):
        self.cache = {}

    def get_orchestrator(self, session_id: str):
        if session_id not in self.cache:
            self.cache[session_id] = Orchestrator(
                index_path=f"faiss_index/{session_id}"
            )

        return self.cache[session_id]


orchestrator_manager = OrchestratorManager()
