import traceback
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.tools.groq_tools import GroqToolClient
from multi_doc_chat.tools.tool_detection import ToolDetector 
from multi_doc_chat.src.document_chat.retrieval import RetrieverWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY


class RAGAgent:
    def __init__(self, orchestrator):
        self.orch = orchestrator

    def run(self, query, chat_history):
        return self.orch.rag_pipeline(query, chat_history)


class ReasoningAgent:
    def __init__(self, orchestrator):
        self.orch = orchestrator

    def run(self, query):
        return self.orch.reason_pipeline(query)


class ToolAgent:
    def __init__(self, orchestrator):
        self.orch = orchestrator

    def run(self, query):
        return self.orch.tool_pipeline(query)


class Orchestrator:
    """
    Holds:
      - RAG pipeline
      - Reasoning pipeline
      - Tool pipeline
      - Chat memory
      - Model loader
      - Retriever
      - Tool detector
    """

    def __init__(self, index_path: str):
        self.model_loader = ModelLoader()

        # ================ Load FAISS ================
        embeddings = self.model_loader.load_embeddings()

        self.vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever_config = self.model_loader.config.get("retriever", {})
        
        # Extract retriever parameters with defaults
        search_type = retriever_config.get("search_type", "mmr")
        k = retriever_config.get("top_k", 10)
        fetch_k = retriever_config.get("fetch_k", 35)
        lambda_mult = retriever_config.get("lambda_mult", 0.45)
        score_threshold = retriever_config.get("score_threshold", 0.45)

        log.info(
            "Initializing Orchestrator",
            index_path=index_path,
            retriever_config=retriever_config
        )

        # Load FAISS Vector Store
        embeddings = self.model_loader.load_embeddings()
        self.vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # Create MMR Retriever
        base_retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
            } if search_type == "mmr" else {"k": k}
        )

        # Wrap with RetrieverWrapper for additional functionality
        self.retriever = RetrieverWrapper(
            retriever=base_retriever,
            search_type=search_type,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold
        )

        # Load tools client
        self.tools_client = GroqToolClient(
            api_keys=[
                self.model_loader.api_key_mgr.get("GROQ_API_KEY_COMPOUND"),
                self.model_loader.api_key_mgr.get("GROQ_API_KEY_DEFAULT"),
            ]
        )

        # Initialize Tool Detector
        self.tool_detector = ToolDetector()

        # Prompts
        self.contextualize_prompt = PROMPT_REGISTRY["contextualize_question"]
        self.qa_prompt = PROMPT_REGISTRY["context_qa"]

        log.info("Orchestrator initialized successfully")

    # Rag Pipeline:
    def rag_pipeline(self, query: str, chat_history):
        try:
            llm = self.model_loader.load_llm("rag")

            log.info("Model loaded for rag with parameters", model=llm)

            # Step 1: rewrite question
            rewritten_query = (
                self.contextualize_prompt
                | llm
                | StrOutputParser()
            ).invoke({"input": query, "chat_history": chat_history})

            log.info("Rewritten query", rewritten_query=rewritten_query)

            docs = self.retriever.retrieve(rewritten_query)
            ctx = "\n\n".join([d.page_content for d in docs])

            # Step 2: Final QA
            answer = (
                self.qa_prompt
                | llm
                | StrOutputParser()
            ).invoke({
                "context": ctx,
                "input": query,
                "chat_history": chat_history
            })

            return {"type": "rag", "content": answer}

        except Exception as e:
            log.error("RAG pipeline error", error=str(e), traceback=traceback.format_exc())
            return {"type": "rag", "content": f"RAG error: {str(e)}"}

    # Reasoning Pipeline:
    def reason_pipeline(self, query: str):
        try:
            llm = self.model_loader.load_llm("reasoning")

            resp = llm.invoke(
                query
            )

            return {
                "type": "reasoning",
                "content": resp.content,
                "reasoning": getattr(resp, "reasoning", None),
            }

        except Exception as e:
            log.error("Reasoning pipeline error", error=str(e))
            return {"type": "reasoning", "content": f"Reasoning error: {str(e)}"}

    # Tool Pipeline:
    def tool_pipeline(self, query: str):
        try:
            tconf = self.model_loader.config["llm"]["tools"]

            result = self.tools_client.call_compound(
                query,
                model = tconf["model_name"],
                enabled_tools = tconf["enabled_tools"],
                max_tokens = tconf.get("max_tokens", 1024),
                stream = False,
            )

            return {"type": "tool", **result}

        except Exception as e:
            log.error("Tool pipeline error", error=str(e))
            return {"type": "tool", "content": f"Tool error: {str(e)}"}
