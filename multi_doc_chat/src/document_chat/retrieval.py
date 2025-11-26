import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any , Tuple

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

    def __init__(self, vectorestore, config):
        self.vectorestore = vectorestore
        self.config = config

        # Last best distance from quick doc-check
        self.last_best_distance: Optional[float] = None

        log.info("RetrieverWrapper initialized with config: " + str(config))

    def quick_relevance_check(self , query:str) -> Tuple[bool, Optional[float]]:
        """
        Quickly checks if the query is relevant to the document corpus.
        Uses a lightweight similarity search to get a relevance score.
        
        Args:
            query: User's query string
        Returns:
            Tuple of (is_query_relevant_to_document: bool, relevance_score: Optional[float] or none)
        """
        try:
            # Fetch the top_k from config for relevance check and default to 3
            top_k_for_check = min(5 , self.config.get("top_k") or 5)

            # Retrieve documents with similarity scores
            docs_with_similarity_scores = self.vectorestore.similarity_search_with_score(
                query,
                k=top_k_for_check
            )

            num_docs = len(docs_with_similarity_scores)

            log.info("Doc-check: retrieved docs", num_docs=num_docs)

            # If no docs found, not relevant set last_best_distance to None and is query relevant to doc to False
            if not docs_with_similarity_scores or num_docs == 0:
                self.last_best_distance = None
                return False, None

            # get the best score by iterating over docs_with_similarity_scores
            scores = [float(s) for _ , s in docs_with_similarity_scores]
            best_distance = min(scores)

            # Assign the value to last best distance
            self.last_best_distance = best_distance

            log.info("Doc-check FAISS distances", scores=scores)
            log.info("Doc-check best distance", best_distance=best_distance)

            # get the threshold value from config
            score_threshold = self.config.get("score_threshold",0.5)
            is_match = best_distance <= score_threshold
            log.info("Doc-check match result", is_relevant_to_document=is_match)

            return is_match, best_distance
        
        except Exception as e:
            log.error(f"Relevance check failed: {e}")
            self.last_best_distance = None
            return False, None

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents using configured search type (MMR by default).
        
        Args:
            query: User's query string
            
        Returns:
            List of relevant Document objects
        """
        try:
            # MMR (Maximal Marginal Relevance) for diversity
            if self.config.get("search_type") == "mmr":
                    
                    log.info("using MMR Search for rtrieval")
                    docs = self.vectorestore.max_marginal_relevance_search(
                        query,
                        k=self.config["top_k"],
                        fetch_k=self.config["fetch_k"],
                        lambda_mult=self.config["lambda_mult"]
                    )
                    log.info("Length of documents retrieved for mmr search - ",num_docs = len(docs))
                    return docs
            
            # Default Similarity
            return self.vectorestore.similarity_search(query, k=self.config["top_k"])
            
        except Exception as e:
            log.error(f"Retrieval failed: {e}")
            return []

