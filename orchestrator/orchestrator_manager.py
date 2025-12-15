# orchestrator/orchestrator_manager.py
from __future__ import annotations
from cachetools import TTLCache
from multi_doc_chat.graph.orchestrator import Orchestrator
from multi_doc_chat.logger import GLOBAL_LOGGER as log


class OrchestratorManager:
    """
    Keeps a per-session cache of Orchestrator instances.

    Each Orchestrator:
      - Loads FAISS index from faiss_index/{session_id}
      - Initializes LLMs, retriever, tools, etc.
    """

    def __init__(self):
        self.cache = TTLCache(maxsize=500, ttl=3600)  # 1 hour

    def get_orchestrator(self, session_id: str) -> Orchestrator:
        """
        Get or lazily create an Orchestrator for a given session.
        """
        if session_id not in self.cache:
            log.info("Creating new Orchestrator | session_id=%s ",session_id)
            self.cache[session_id] = Orchestrator(
                index_path=f"faiss_index/{session_id}"
            )
        else:
            log.debug("Reusing cached Orchestrator| session_id=%s ", session_id)

        return self.cache[session_id]


orchestrator_manager = OrchestratorManager()
