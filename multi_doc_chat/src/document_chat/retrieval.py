import math
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.model_loader import ModelLoader


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
        self, vectorestore, model_loader: ModelLoader, retriever_config, reranker_config
    ):
        # passed via orchestrator during initialization of Retriever
        self.vectorestore = vectorestore
        self.model_loader = model_loader
        self.retriever_config = retriever_config or {}
        self.reranker_config = reranker_config or {}

        # Loaded once in ModelLoader.__init__ and get the reranker model
        self.reranker = model_loader.get_reranker()

        # Last best distance from quick doc-check
        self.last_best_distance: Optional[float] = None

        log.info(
            "RetrieverWrapper initialized",
            retriever_cfg=self.retriever_config,
            reranker_config=self.reranker_config,
            reranker_enabled=bool(self.reranker),
        )

    def quick_relevance_check(self, query: str) -> Tuple[bool, Optional[float]]:
        """
        Quickly checks if the query is relevant to the document corpus.
        Uses a lightweight similarity search to get a relevance score.

        Args:
            query: User's query string
        Returns:
            Tuple of (is_query_relevant_to_document: bool, relevance_score: float or none)
        """
        try:
            top_k_for_check = self.reranker_config.get("top_k_routing", 10)

            # Retrieve documents with similarity scores - top_k_for_check
            docs_with_scores = self.vectorestore.similarity_search_with_score(
                query, k=top_k_for_check
            )

            num_docs = len(docs_with_scores)

            log.info("Doc-check: retrieved docs", num_docs=num_docs)

            # If no docs found, not relevant set last_best_distance to None and is query relevant to doc to False
            if not docs_with_scores or num_docs == 0:
                self.last_best_distance = None
                return False, None

            faiss_scores = [float(s) for _, s in docs_with_scores]
            best_faiss = min(faiss_scores)

            # converting the best faiss distance to similarity:
            faiss_sim = 1 / (1 + best_faiss)

            log.info("fAISS normalized similarity score : ", faiss_sim=faiss_sim)

            log.info(
                "Doc-check (FAISS)",
                list_of_all_scores=faiss_scores,
                best_score=best_faiss,
            )

            if self.reranker:
                log.info("Applying the reranker logic")

                # Creating the pairs of query and docs fetched from faiss
                pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]
                # Passing it to the reranker model and getting the reranked scores
                rerank_scores = self.reranker.predict(pairs, batch_size=8)

                # convert numpy types:
                rerank_scores = [float(s) for s in rerank_scores]
                # getting the best reranked scored:
                best_rerank = max(rerank_scores)

                log.info(
                    "Doc-check (Reranker)",
                    reranked_scores=rerank_scores,
                    best_reranked_Score=best_rerank,
                )

                # Convert reranker output to normalized similarity
                rerank_sim = 1 / (1 + math.exp(-best_rerank))
                log.info(
                    "Rerankers normalized similarit score : ", rerank_sim=rerank_sim
                )

            # Weighted combination of faiss and reranker scores
            alpha = self.reranker_config.get("faiss_weight", 0.6)
            beta = self.reranker_config.get("rerank_weight", 0.4)

            final_score = alpha * faiss_sim + beta * rerank_sim
            log.info(
                "Final score after combining normaloized faiss and rerank score: ",
                final_score=final_score,
            )

            # Save for router
            self.last_best_distance = final_score

            is_relevant = final_score >= 0.56

            # If no reranker installed → fallback to FAISS thresholding
            # log.info("Reranked disabled - applying basic faiss thresholding logic")
            # is_relevant = best_faiss <= self.retriever_config.get(
            #     "score_threshold", 0.55
            # )
            # self.last_best_distance = best_faiss
            return is_relevant, final_score

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
            fetch_k = self.reranker_config.get("top_k_retrieval")  # 25
            final_k = self.reranker_config.get("final_k")  # 6

            fetch_k_mmr = self.retriever_config.get("fetch_k", 35)  # 35
            top_k_for_mmr = self.retriever_config.get("top_k", 10)  # 8

            # MMR (Maximal Marginal Relevance) for diversity
            if self.retriever_config.get("search_type") == "mmr":
                lambda_mult = self.retriever_config.get("lambda_mult", 0.5)

                log.info(
                    "using MMR Search for retrieval with config parameters",
                    final_k_sent_to_RAG_LLM=top_k_for_mmr,
                    fetch_k=fetch_k_mmr,
                    lambda_mult=lambda_mult,
                )

                docs = self.vectorestore.max_marginal_relevance_search(
                    query, k=top_k_for_mmr, fetch_k=fetch_k_mmr, lambda_mult=lambda_mult
                )
                log.info(
                    "Length of documents retrieved for mmr search - ",
                    num_docs=len(docs),
                )
            else:
                # if search type is not set to mmr then retrieve using basic similarity search
                docs = self.vectorestore.similarity_search(query, k=fetch_k)

            # Reranking the retrieved docs from mmm/similarity searchid enabled:
            pairs = [(query, d.page_content) for d in docs]
            scores = self.reranker.predict(pairs)
            scores = [float(s) for s in scores]

            # create a reranked list of docs and corresponding scores (sort them in descending order)
            reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

            # Get the top 5 reranked docs:
            final_docs = [d for d, s in reranked[:final_k]]

            log.info("Final reranked retrieval", final_count=len(final_docs))

            return final_docs
        except Exception as e:
            log.error(f"Retrieval failed: {e}")
            return []

    def return_docs_from_ids(self, ids: List[str]) -> List[Document]:
        """
        Reconstruct Document objects from stored IDs in FAISS docstore.

        Assumes:
        - During data ingestion , metadata["id"] was assigned to each doc
        - FAISS was built with ids = metadata["id"] , so docstore keys match
        """
        docs: List[Document] = []
        store = self.vectorestore.docstore

        # Iterate over each doc-id from the ids list
        for _id in ids:
            try:
                # get the doc respective to id from the docstore
                doc = store[_id] 
                # if a doc is found append it to list of "docs" 
                if doc:
                    docs.append(doc)
            except KeyError:
                log.warning("Doc ID not found in FAISS docstore", doc_id=_id)
            except Exception as e:
                log.warning("Failed to load doc from id", doc_id=_id, error=str(e))

        log.info("docs_from_ids result", num_docs=len(docs))
        return docs

    def embed_query(self, query: str) -> List[float]:
        """
        Using the same embedding function FAISS was built with to embed a query.
        This is used for:
          - Semantic retrieval
          - Semantic retrieval-cache matching
        """
        try:
            embd_func = getattr(self.vectorestore, "embedding_function", None)
            if embd_func is None:
                raise ValueError("Vectorstore has no embedding_function")
            # return the embedded query:
            return embd_func.embed_query(query)
        except Exception as e:
            log.error(f"Failed to embed query: {e}")
            raise
