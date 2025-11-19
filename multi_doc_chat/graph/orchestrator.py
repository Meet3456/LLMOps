import traceback
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.tools.groq_tools import GroqToolClient
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

        self.retriever = RetrieverWrapper(
            self.vectorstore.as_retriever(search_kwargs={"k": 10})
        )

        # ================ Load tools client =========
        self.tools_client = GroqToolClient(
            api_key=self.model_loader.api_keys.get("GROQ_API_KEY")
        )

        # ================ Prompts ===================
        self.contextualize_prompt = PROMPT_REGISTRY["contextualize_question"]
        self.qa_prompt = PROMPT_REGISTRY["context_qa"]

    # =====================================================================
    #                        RAG PIPELINE
    # =====================================================================
    def rag_pipeline(self, query: str, chat_history):
        try:
            llm = self.model_loader.load_llm("rag")

            # Step 1: rewrite question
            rewritten = (
                {"input": query, "chat_history": chat_history}
                | self.contextualize_prompt
                | llm
                | StrOutputParser()
            ).invoke({"input": query, "chat_history": chat_history})

            docs = self.retriever.retrieve(rewritten)
            ctx = "\n\n".join([d.page_content for d in docs])

            # Step 2: Final QA
            answer = (
                {
                    "context": ctx,
                    "input": query,
                    "chat_history": chat_history,
                }
                | self.qa_prompt
                | llm
                | StrOutputParser()
            ).invoke(
                {"context": ctx, "input": query, "chat_history": chat_history}
            )

            return {"type": "rag", "content": answer}

        except Exception as e:
            log.error("RAG pipeline error", error=str(e), traceback=traceback.format_exc())
            return {"type": "rag", "content": f"RAG error: {str(e)}"}

    # =====================================================================
    #                        REASONING PIPELINE
    # =====================================================================
    def reason_pipeline(self, query: str):
        try:
            llm = self.model_loader.load_llm("reasoning")

            resp = llm.invoke(
                query,
                include_reasoning=True,
                reasoning_effort="high",
            )

            return {
                "type": "reasoning",
                "content": resp.content,
                "reasoning": getattr(resp, "reasoning", None),
            }

        except Exception as e:
            log.error("Reasoning pipeline error", error=str(e))
            return {"type": "reasoning", "content": f"Reasoning error: {str(e)}"}

    # =====================================================================
    #                        TOOL PIPELINE
    # =====================================================================
    def tool_pipeline(self, query: str):
        try:
            tconf = self.model_loader.config["llm"]["tools"]

            result = self.tools_client.call_compound(
                query,
                model=tconf["model_name"],
                enabled_tools=tconf["enabled_tools"],
                reasoning_format=tconf.get("reasoning_format", "parsed"),
                include_reasoning=tconf.get("include_reasoning", True),
                stream=False,
            )

            return {"type": "tool", **result}

        except Exception as e:
            log.error("Tool pipeline error", error=str(e))
            return {"type": "tool", "content": f"Tool error: {str(e)}"}
