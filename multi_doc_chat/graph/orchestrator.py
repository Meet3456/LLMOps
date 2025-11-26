import json
import traceback
from typing import List

from pydantic import BaseModel, Field
from typing_extensions import Literal

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.src.document_chat.retrieval import RetrieverWrapper
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.tools.groq_tools import GroqToolClient
from multi_doc_chat.exception.custom_exception import DocumentPortalException


# Route query class
class RouteQuery(BaseModel):
    source: Literal["rag", "tools", "reasoning"] = Field(
        ... , description="Which agent should handle this query."
    )

class Orchestrator:
    """
    Orchestrates:
      - LLM-based routing (router LLM)
      - RAG pipeline
      - Reasoning pipeline
      - Tool pipeline (Groq Compound)
      - Retrieval
    """

    def __init__(self, index_path: str):
        # Create an instance of ModelLoader class:
        self.model_loader = ModelLoader()
        # as is ml we call = "self.config = load_config()" , so we can save it
        self.config = self.model_loader.config

        # Retrieval
        self._init_retriever(index_path)
        # LLMs
        self._init_models()
        # Tools (compound)
        self._init_tools()

        log.info("Orchestrator initialized successfully")


    # Function which initializes retriever:
    def _init_retriever(self, index_path:str):
        # load embedddings
        embeddings = self.model_loader.load_embeddings()

        # Load vectorestore(faiss local)
        vectorestore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # get the retriever config(which is mmr , can be changed as per req.)
        retriever_confg = self.config.get("retriever", {})

        # get the retriever object which is returned by the RetrieverWrapper Class 
        # retriever attribute has 2 methods - {"quick_relevance_check" and "retrieve"}
        self.retriever = RetrieverWrapper(vectorestore=vectorestore , config=retriever_confg)
        log.info("Retriever initialized successfully")


    # Function which initializes llms
    def _init_models(self):
        # Router llm
        router_llm_raw = self.model_loader.load_llm("router")
        self.router_prompt = PROMPT_REGISTRY["router"]
        self.router_llm = router_llm_raw.with_structured_output(RouteQuery)

        # RAG LLM
        self.rag_llm = self.model_loader.load_llm("rag")
        self.contextualize_prompt = PROMPT_REGISTRY["contextualize_question"]
        self.qa_prompt = PROMPT_REGISTRY["context_qa"]

        # Reasoning LLM
        self.reasoning_llm = self.model_loader.load_llm("reasoning")
        log.info("All llm's initialized successfully")


    # gets api keys from model loader and initializes GrogToolCient:
    def _init_tools(self):
        self.tools_client = GroqToolClient(
            api_keys=[
                self.model_loader.api_key_mgr.get("GROQ_API_KEY_COMPOUND"),
                self.model_loader.api_key_mgr.get("GROQ_API_KEY_DEFAULT"),
            ]
        )
        log.info("Tools llm initialized successfully")
        
    
    def _built_routing_signals(self , query:str):
        q_lower = query.lower()

        # Doc-check via FAISS , quick_relevance_check = returns a boolean whether is_query_relevant_to_document and the relevance_scorem distance(minimum)
        is_query_relevant_to_document , best_distance = self.retriever.quick_relevance_check(query)

        contains_url = "http://" in q_lower or "https://" in q_lower

        contains_math = any(ch in q_lower for ch in "+-*/=") and any(
            w in q_lower for w in ["solve", "calculate", "compute", "evaluate"]
        )
        asks_for_latest = any(
            kw in q_lower for kw in ["latest", "today", "current",]
        )

        token_approx = len(q_lower.split())

        signals = {
            "doc_match": is_query_relevant_to_document,
            "best_distance": best_distance,
            "contains_url": contains_url,
            "contains_math": contains_math,
            "asks_for_latest": asks_for_latest,
            "approx_tokens": token_approx,
        }

        log.info("Routing signals", signals=signals)
        return signals