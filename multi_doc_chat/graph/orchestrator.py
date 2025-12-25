import json
import traceback
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing_extensions import Literal

from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.graph.builder import build_graph
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.src.document_chat.retrieval import RetrieverWrapper
from multi_doc_chat.tools.groq_tools import GroqToolClient
from multi_doc_chat.utils.model_loader import ModelLoader


# Route query class
class RouteQuery(BaseModel):
    source: Literal["rag", "tools", "reasoning"] = Field(
        ..., description="Which agent should handle this query."
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

        # compile the graph once at initialization(build_graph retruns the compoiled graph)
        self.graph = build_graph()

        # we pass the index path in orchestrator manager:
        log.info("Orchestrator initialized | index_path=%s", index_path)

    # Function which initializes retriever:
    def _init_retriever(self, index_path: str):
        # load embedddings
        embeddings = self.model_loader.load_embeddings()

        # Load vectorestore(faiss local)
        vectorestore = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )

        # get the retriever config(which is mmr , can be changed as per req):
        retriever_config = self.config.get("retriever", {})

        # get the reranker model config:
        reranker_cfg = self.config.get("reranker", {})

        # get the retriever object which is returned by the RetrieverWrapper Class
        # retriever attribute has 2 methods - {"quick_relevance_check" and "retrieve"}
        self.retriever = RetrieverWrapper(
            vectorestore=vectorestore,
            model_loader=self.model_loader,
            retriever_config=retriever_config,
            reranker_config=reranker_cfg,
        )
        log.info("Retriever initialized successfully | index_path=%s", index_path)

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

    # Builds signals(bases on user query) for helping llm for routing
    def _built_routing_signals(self, query: str):
        try:
            q_lower = query.lower()

            # Doc-check via FAISS , quick_relevance_check = returns a boolean whether is_query_relevant_to_document and the relevance_scorem distance(minimum)
            is_query_relevant_to_document, best_distance = (
                self.retriever.quick_relevance_check(query)
            )

            contains_url = "http://" in q_lower or "https://" in q_lower

            contains_math = any(ch in q_lower for ch in "+-*/=") and any(
                w in q_lower for w in ["solve", "calculate", "compute", "evaluate"]
            )
            asks_for_latest = any(
                kw in q_lower
                for kw in [
                    "latest",
                    "today",
                    "current",
                    "recent",
                    "recently",
                    "who won",
                    "scoreline",
                    "news",
                    "update",
                ]
            )

            token_approx = len(q_lower.split())

            signals = {
                "query_related_to_fetched_documents": is_query_relevant_to_document,
                "best_distance": best_distance,
                "contains_url": contains_url,
                "contains_math": contains_math,
                "asks_for_latest": asks_for_latest,
                "approx_tokens": token_approx,
            }

            log.info(f"Routing signals | signals = {signals}")
            return signals
        except Exception as e:
            log.error(f"Failed to generate routing signals | error = {str(e)}")
            raise DocumentPortalException(
                "Error in genereating routing signals", e
            ) from e

    # Based on signals and router llm retruns the routed node:
    def route_query(self, query: str, chat_history: List):
        """
        The following function uses router llm + routing signlas to pick route
        - "rag","reasoning","tools"
        """
        try:
            # Built the signals based on the user query:
            signals = self._built_routing_signals(query)

            if signals["query_related_to_fetched_documents"]:
                return "rag"

            if signals["asks_for_latest"] or signals["contains_url"]:
                return "tools"

            # pass to llm if not clear
            chain = self.router_prompt | self.router_llm

            # Invoke the chain with necessary inputs
            result = chain.invoke({"input": query, "signals": json.dumps(signals)})
            log.info("Langchain llm result | langchain_result=%s", result)

            selected_route = result.source
            log.info(f"Fetching the llm result | route_selected={selected_route}")

            return selected_route
        except Exception as e:
            log.error("Router failed | error=%s", str(e))
            return "reasoning"

    # Function which runs the rag pipeline(if rourted to rag node):
    def run_rag(
        self,
        query: str,
        chat_history: List,
        docs: Optional[List[Document]] = None,
        skip_retrieval: bool = False,
    ):
        try:
            # Initialize the rag llm:
            llm = self.rag_llm

            # Rewrite the user query wrt to chat_history(if present)
            if chat_history:
                rewrite_query_chain = (
                    self.contextualize_prompt | llm | StrOutputParser()
                )

                rewritten_query = rewrite_query_chain.invoke(
                    {"input": query, "chat_history": chat_history}
                )
                log.info(
                    f"users input query successfully rewritten based on prev chat history | rewritten_query={rewritten_query}"
                )
            else:
                log.info("no chat_history passing default user input query")
                rewritten_query = query

            if skip_retrieval:
                log.info(
                    "run_rag | retrieval skipped (cache hit) Docs already fetched from Cache"
                )
            else:
                # If docs are not supplied → do normal retrieval
                if docs is None:
                    # Retrieve relevant docs with mmr search from the faiss index
                    docs = self.retriever.retrieve(rewritten_query)

            if not docs or len(docs) == 0:
                log.info("RAG: no documents retrieved")
                return (
                    "I don't know based on the available documents. "
                    "I could not find any relevant content in the knowledge base."
                )
            log.info("Length of retrieved docs for RAG | docs=%d", len(docs))

            # Fetch the content from docs and builf the context:
            context = "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

            # Final qa chain:
            qa_chain = self.qa_prompt | llm | StrOutputParser()

            answer = qa_chain.invoke(
                {
                    "context": context,
                    "input": rewritten_query,
                    "chat_history": chat_history,
                }
            )
            return answer

        except Exception as e:
            log.error(
                "RAG failed | error=%s | traceback=%s",
                str(e),
                traceback.format_exc(),
            )
            raise DocumentPortalException("RAG pipeline failed", e)

    # Function ehich runs the reasoning pipeline(if routed to reasoning node)
    def run_reasoning(self, query: str):
        try:
            resp = self.reasoning_llm.invoke([{"role": "user", "content": query}])

            # Extract final answer
            content = getattr(resp, "content", "") or ""

            # Extract reasoning (Qwen stores it inside response_metadata)
            if hasattr(resp, "additional_kwargs"):
                reasoning = resp.additional_kwargs.get("reasoning_content")

            log.info("Reasoning response | has_reasoning=%s ", reasoning)

            # Return combined or separate depending on your UI design
            if reasoning:
                return f"{content}\n\n---\n🧠 Reasoning:\n{reasoning}"
            return content

        except Exception as e:
            log.error("Reasoning failed | error=%s", str(e))
            return "Reasoning failed."

    # Function ehich runs the tool pipeline(if routed to tool node)
    def run_tools(self, query: str):
        """
        Calls Groq Compound Mini through GroqToolClient.
        """
        try:
            tconf = self.config["llm"]["tools"]

            result = self.tools_client.call_compound(
                user_prompt=query,
                model=tconf["model_name"],
                enabled_tools=tconf["enabled_tools"],
                max_tokens=tconf.get("max_tokens", 1024),
                stream=False,
            )
            log.info("Successfully got tools output")
            return result.get("content", "")

        except Exception as e:
            log.error("Tools failed | error=%s", str(e))
            return "Tool execution failed."
