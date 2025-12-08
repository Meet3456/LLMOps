import os
import sys

import torch
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder

from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.config_loader import load_config


class ApiKeyManager:
    REQUIRED = ["GROQ_API_KEY_DEFAULT", "GROQ_API_KEY_COMPOUND", "GOOGLE_API_KEY"]

    def __init__(self):
        load_dotenv()
        self.keys = {}

        # Iterate over the required keys:
        for k in self.REQUIRED:
            # get the value of the specific key from Required list from Env variables:
            if val := os.getenv(k):
                # Assign the value to respective key in keys dict:
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
        # Initialize API Key Manager - as we create the object of the class it will load and validate Env API Keys and store it in keys attribute:
        self.api_key_mgr = ApiKeyManager()

        # Store all API keys in a single attribute for easy access : as keys contains all required keys
        self.api_keys = self.api_key_mgr.keys

        # Load configuration
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

        self.reranker = self._load_reranker()

    # Loads reranker overall once into system:
    def _load_reranker(self):
        """Load reranker only if enabled."""
        rerank_cfg = self.config.get("reranker", {})

        if not rerank_cfg.get("enabled", False):
            log.info("Reranker disabled.")
            return None

        model_name = rerank_cfg.get("model_name", "BAAI/bge-reranker-base")

        dtype = torch.float32

        log.info(f"Loading RERANKER: {model_name} using dtype={dtype}")

        model = CrossEncoder(
            model_name,
            model_kwargs={"dtype": dtype},
            max_length=512,
        )

        log.info("Reranker loaded successfully")
        return model

    # will not download the model again and again instead gets it from the system:
    def get_reranker(self):
        """Return the loaded reranker (or None)."""
        return self.reranker

    def load_embeddings(self):
        """
        Load and return embedding model from Google Generative AI.
        """
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)
            return GoogleGenerativeAIEmbeddings(
                model=model_name, google_api_key=self.api_keys.get("GOOGLE_API_KEY")
            )
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)

    # logic to select groq api key based on role
    def _select_groq_key(self, role: str) -> str:
        """
        Select appropriate Groq API key based on role.
        Args:
            role: One of "rag", "reasoning", or "tools"

        Returns:
            API key string
        """
        if role == "tools":
            return self.api_keys.get("GROQ_API_KEY_COMPOUND")
        return self.api_keys.get("GROQ_API_KEY_DEFAULT")

    # load role-based llm:
    def load_llm(self, role: str):
        """
        Load and return the configured LLM model.
        Args:
            role: One of "rag", "reasoning", or "tools"

        Returns:
            Configured LLM instance
        """
        # role in {"rag", "reasoning", "tools"}
        if role not in self.config["llm"]:
            log.error("LLM role not found in config", role=role)
            raise ValueError(f"LLM role '{role}' not found in config")

        # get the respectieve llm config for specified role:
        llm_config = self.config["llm"][role]

        provider = llm_config["provider"]
        model = llm_config["model_name"]
        temp = llm_config.get("temperature")
        max_t = llm_config.get("max_tokens")

        log.info(f"Loading LLM for role={role}", model=model)

        if provider == "google":
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=self.api_keys.get("GOOGLE_API_KEY"),
                temperature=temp,
                max_output_tokens=max_t,
            )

        if provider == "groq":
            api_key = self._select_groq_key(role)

            if role == "reasoning":
                log.info("Loading reasoning model")
                reasoning_effort = llm_config.get("reasoning_effort")
                reasoning_format = llm_config.get("reasoning_format")
                top_p = llm_config.get("top_p")

                # GPT-OSS *must not* receive model_kwargs
                return ChatGroq(
                    model=model,
                    api_key=api_key,
                    temperature=temp,
                    max_tokens=max_t,
                    reasoning_effort=reasoning_effort,  # low|medium|high
                    reasoning_format=reasoning_format,  # true/false
                    model_kwargs={"top_p": top_p} if top_p is not None else None,
                )

            model_kwargs = {}

            top_p = llm_config.get("top_p")
            freq_pen = llm_config.get("frequency_penalty")
            pres_pen = llm_config.get("presence_penalty")

            if top_p is not None:
                model_kwargs["top_p"] = top_p
            if freq_pen is not None:
                model_kwargs["frequency_penalty"] = freq_pen
            if pres_pen is not None:
                model_kwargs["presence_penalty"] = pres_pen

            # specifically for rag llm
            if model_kwargs:
                return ChatGroq(
                    model=model,
                    api_key=api_key,
                    temperature=temp,
                    max_tokens=max_t,
                    model_kwargs=model_kwargs,  # safe
                )

            # default groq llm loading(router,tools)
            return ChatGroq(
                model=model,
                api_key=api_key,
                temperature=temp,
                max_tokens=max_t,
            )

        raise ValueError(f"Unsupported provider {provider}")
