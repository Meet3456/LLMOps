"""
Docstring for workflow:

When new query is enetered by the user:

query
→ embedding
→ FAISS search
→ reranker
→ MMR
→ final_docs = [doc1, doc2, doc3...]
→ cache above final doc_ids
→ run RAG answer
→ cache final answer

Next (paraphrased) time:

→ check answer cache         (fast)
→ check retrieval cache      (fast)
→ docs_from_ids()           (fast)
→ run RAG on cached docs    (fast)


"""
