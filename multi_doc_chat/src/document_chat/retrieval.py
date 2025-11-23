import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document

from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log


class RetrieverWrapper:
    """
    Wraps FAISS retriever with MMR support and performs:
     - Query relevance detection
     - MMR-based document retrieval (balances relevance + diversity)
     - Configurable search parameters
    
    MMR (Maximal Marginal Relevance):
    - Reduces redundancy in retrieved documents
    - lambda_mult controls diversity vs relevance tradeoff
        * 0.0 = maximum diversity (completely different docs)
        * 1.0 = maximum relevance (might be similar docs)
        * 0.5 = balanced approach (recommended)
    """

    def __init__(
        self,
        retriever,
        search_type,
        k,
        fetch_k,
        lambda_mult,
        score_threshold
    ):
        self.retriever = retriever
        self.search_type = search_type
        self.k = k
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
        self.score_threshold = score_threshold


    def is_document_query(self, query: str) -> bool:
        """
        Simple heuristic: if the retriever returns ANY documents → it's a doc query.
        """

        try:
            log.info("threshold check", score_threshold=self.score_threshold)

            # FAISS distance-based retrieval
            docs_with_scores = self.retriever.vectorstore.similarity_search_with_score(
                query,
                k = self.k
            )

            log.info("Docs retrieved", num_docs=len(docs_with_scores))

            if not docs_with_scores:
                log.info("No docs returned by FAISS")
                return False

            docs, scores = zip(*docs_with_scores)

            # Ensure conversion to float
            scores = [float(s) for s in scores]

            log.info("FAISS distances", scores=scores)

            # LOWER = better
            best_distance = min(scores)

            log.info("Best distance", value=best_distance)

            # Correct threshold logic
            if best_distance <= self.score_threshold:
                log.info("RAG document match", distance=best_distance)
                return True

            log.info("Distance above threshold → NOT a doc query", distance=best_distance)
            return False
        except Exception as e:
            log.warning("retriever failed", error=str(e))
            return False

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents using configured search type (MMR by default).
        
        Args:
            query: User's query string
            
        Returns:
            List of relevant Document objects
        """
        try:
            log.info("Starting retrieval", search_type=self.search_type)

            if self.search_type == "mmr":
                # MMR retrieval: balances relevance and diversity
                docs = self.retriever.get_relevant_documents(
                    query,
                    k=self.k,
                    fetch_k=self.fetch_k,
                    lambda_mult=self.lambda_mult
                )
                log.info(
                    "MMR retrieval completed",
                    query=query[:50],
                    num_docs=len(docs),
                    fetch_k=self.fetch_k,
                    lambda_mult=self.lambda_mult
                )
                
            elif self.search_type == "similarity":
                # Standard similarity search
                docs = self.retriever.get_relevant_documents(query, k=self.k)
                log.info(
                    "Similarity retrieval completed",
                    query=query[:50],
                    num_docs=len(docs)
                )
                
            elif self.search_type == "similarity_score_threshold":
                # Only return docs above score threshold
                docs = self.retriever.get_relevant_documents(
                    query,
                    k=self.k,
                    score_threshold=self.score_threshold
                )
                log.info(
                    "Threshold retrieval completed",
                    query=query[:50],
                    num_docs=len(docs),
                    threshold=self.score_threshold
                )
            else:
                raise ValueError(f"Unknown search type: {self.search_type}")
            
            return docs
            
        except Exception as e:
            log.error("Retrieval failed", error=str(e), query=query[:50])
            raise DocumentPortalException("Document retrieval error", e)