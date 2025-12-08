┌─────────────────────────────────────────────────────────────────────────┐
│                           USER (Streamlit UI)                           │
│   - Upload documents                                                   │
│   - Ask questions                                                      │
│   - Sees chat history per session                                      │
└─────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     FASTAPI — /upload  (Create Session)                │
│  - ChatRepository.create_session() writes session → DB                 │
│  - DataIngestor(session_id) launched                                   │
└─────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────── DOCUMENT INGESTION PIPELINE ────────────────────┐
│ DataIngestor                                                            │
│   1) Save uploaded files → data/{session_id}/                           │
│   2) load_documents_and_assets():                                       │
│        - PDFs → text, tables, images, captions                          │
│        - DOCX/TXT → text                                                │
│        - CSV/HTML → table + text                                        │
│        - Images → caption                                               │
│        => Each Document gets metadata:                                  │
│           { modality, source, page?, table_json?, image_path?, ... }    │
│   3) _multimodal_split():                                               │
│        - Split text chunks                                              │
│        - Split table chunks                                             │
│        - Keep images unsplit                                            │
│   4) Assign stable chunk IDs → metadata["id"]                           │
│        Example: session_18_nov_2025_3pm_d1e5__42_8a31f0c2               │
│   5) FAISS Manager                                                      │
│        - load_or_create_index()                                         │
│        - add_documents(): inserts docs with IDs                         │
│        => FAISS docstore keys == metadata["id"]                         │
└─────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌────────────────────────────── VECTORSTORE ─────────────────────────────┐
│ FAISS Index (session-scoped)                                           │
│   Stored under: faiss_index/{session_id}/                              │
│   Contains:                                                            │
│     - embeddings of each chunk                                         │
│     - docstore keyed by stable IDs                                     │
│     - metadata persisted                                                │
└─────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  FASTAPI — /chat  (Query Flow)                         │
│  1. Normalize user query                                                │
│  2. Check Redis Answer Cache                                            │
│       cache_answer(session_id, query_norm) → answer?                   │
│  3. Check Redis Retrieval Cache                                         │
│       lookup_retrieval_entry(session_id, query_norm) → doc_ids?        │
│  4. If doc_ids cached → docs_from_ids(docstore)                         │
│  5. Else → FAISS Retrieval:                                             │
│       retriever.retrieve():                                            │
│          - embedding                                                    │
│          - FAISS kNN/MMR                                                │
│          - reranker                                                     │
│       -> final_docs, store doc_ids in Redis                             │
│  6. Build chat history from DB                                          │
│  7. Use Orchestrator: route → rag/tool/reasoning                        │
│  8. Generate answer → cache_answer                                      │
│  9. Save messages → Postgres                                            │
└─────────────────────────────────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                               Streamlit UI                             │
│   - Shows conversation                                                  │
│   - Shows multiple sessions                                             │
│   - Sends new queries to FastAPI                                        │
└─────────────────────────────────────────────────────────────────────────┘
