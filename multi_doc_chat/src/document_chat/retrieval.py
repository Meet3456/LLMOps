import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.models.model import PromptType, ChatAnswer
from pydantic import ValidationError


class RetrieverWrapper:
    """
    Wraps FAISS retriever and performs:
     - Query relevance detection
     - Query → documents
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def is_document_query(self, query: str) -> bool:
        """
        Simple heuristic: if the retriever returns ANY documents → it's a doc query.
        """

        try:
            docs = self.retriever.get_relevant_documents(query)
            if docs and len(docs) > 0:
                return True
        except Exception as e:
            log.warning("retriever failed", error=str(e))

        return False

    def retrieve(self, query: str):
        return self.retriever.get_relevant_documents(query)