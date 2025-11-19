import os
import sys
import json
from dotenv import load_dotenv
from multi_doc_chat.utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException



class ApiKeyManager:
    REQUIRED = ["GROQ_API_KEY", "GOOGLE_API_KEY"]

    def __init__(self):
        load_dotenv()
        self.keys = {}

        for k in self.REQUIRED:
            if val := os.getenv(k):
                self.keys[k] = val
                log.info(f"Loaded {k} from env")
            else:
                log.error(f"Missing required API key: {k}")

        if len(self.keys) != len(self.REQUIRED):
            raise DocumentPortalException("Missing API Keys", sys)

    def get(self, key: str) -> str:
        return self.keys[key]


class ModelLoader:
    """
    Responsible for:
    - Loading embeddings
    - Loading the RAG LLM
    - Loading the Reasoning LLM
    - Loading the Tool LLM (compound / compound-mini)
    """

    def __init__(self):
        # Initialize API Key Manager - as we create the object of the class it will load and validate Env API Keys:
        self.api_key_mgr = ApiKeyManager()

        # Load configuration
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))


    def load_embeddings(self):
        """
        Load and return embedding model from Google Generative AI.
        """
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)
            return GoogleGenerativeAIEmbeddings(model=model_name,
                                                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY")) #type: ignore
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    def load_llm(self, role: str):
        """
        Load and return the configured LLM model.
        """
        # role in {"rag", "reasoning", "tools"}
        if role not in self.config["llm"]:
            log.error("LLM role not found in config", role=role)
            raise ValueError(f"LLM role '{role}' not found in config")

        # get the respectieve llm config for specified role:
        llm_config = self.config["llm"][role]
        provider = llm_config["provider"]
        model = llm_config["model_name"]
        temp = llm_config.get("temperature", 0.3)
        max_t = llm_config.get("max_tokens", 2048)

        log.info(f"Loading LLM for role={role}", model=model)

        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=self.api_keys.get("GOOGLE_API_KEY"),
                temperature=temp,
                max_output_tokens=max_t,
            )

        if provider == "groq":
            return ChatGroq(
                model=model,
                api_key=self.api_keys.get("GROQ_API_KEY"),
                temperature=temp,
                max_tokens=max_t,
                # reasoning params passed at call time
            )

        raise ValueError(f"Unsupported provider {provider}")


if __name__ == "__main__":
    loader = ModelLoader()

    # Test Embedding
    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")
    result = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Result: {result}")

    # Test LLM
    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")
    result = llm.invoke("Hello, how are you?")
    print(f"LLM Result: {result.content}")