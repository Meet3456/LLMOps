"""
FastAPI runs on Asyncio - so the asyncio event Loop should never be blocked

What is Event Loop?
    - event loop is as: A single smart manager - {who coordinates many tasks, but does not do heavy labor}
    - It switches between tasks extremely fast
    - It waits for I/O
    - Schedules Work

Event Loop:
    - User A: waiting for PDF parsing
    - User B: waiting for image caption
    - User C: waiting for DB write

Note - Blocking operations should not be loaded into Event loop(as they are computation Heavy) - The event Loop Freezes
| Operation        | Why blocking      |
| ---------------- | ----------------- |
| PDF parsing      | CPU heavy         |
| FAISS indexing   | CPU heavy         |
| Disk reads       | OS-level blocking |
| Image processing | CPU + memory      |

Non-Blocking Operations:
| Operation             | Why safe       |
| --------------------- | -------------- |
| await network I/O     | OS handles     |
| await asyncio.sleep   | yields control |
| await DB async client | cooperative    |


------------------------------------------------------------------------------------------------------------------------------


THREADPOOL EXECUTOR (THE WORKERS)

- What is ThreadPoolExecutor?
    executor = ThreadPoolExecutor(max_workers=16)

- Means:
    “I have 16 background workers to do heavy jobs”

- These workers:
    - run blocking code
    - do CPU & disk work (spread te work between 16 workers)
    - do not block the event loop

- Key idea
    - Event loop delegates heavy work
    - Threadpool executes heavy work


------------------------------------------------------------------------------------------------------------------------------


run_in_executor() (THE BRIDGE)

- Syntax
    await loop.run_in_executor(executor, func, arg1, arg2)

- Meaning:

    “Hey event loop,
    please ask a worker thread to run `func(arg1, arg2)`
    and notify me when it’s done.”

- What happens internally?

    - Event loop submits job to executor
    - Executor assigns a free worker thread
    - Worker thread - {runs blocking code}
    - Event loop continues serving others {in background the task is assigned to a worker thread}
    - Result comes back
    - await resumes


------------------------------------------------------------------------------------------------------------------------------


- In document_ops.py :
    Threadpool creation
    executor = ThreadPoolExecutor(max_workers=16)
    So -> we have 16 workers.

- _process_single_path :
    async def _process_single_path(p, images_dir, tables_dir):
        loop = asyncio.get_running_loop()

        text_docs = await loop.run_in_executor(
            executor, loader.load
        )

- What is happening?
    Step	    Description
     1	    Event loop starts _process_single_path
     2	    PDF parsing is detected
     3	    Parsing is delegated(moved towards) to threadpool
     4	    Event loop moves to next task (in background the threads spawned do the Parsing task)
     5	    Worker parses PDF
     6	    Result comes back
     7	    Coroutine(await) resumes

- asyncio.gather
    tasks = [_process_single_path(p) for p in paths]
    results = await asyncio.gather(*tasks)

    Meaning:
    - “Process all files concurrently”
    - All the files (which are present in paths list are processed Together), parallely/concurrently


------------------------------------------------------------------------------------------------------------------------------


One user uploads MULTIPLE PDFs - Upload: [A.pdf, B.pdf, C.pdf]

- STEP: 1
    asyncio.gather(
      _process_single_path(A),
      _process_single_path(B),
      _process_single_path(C)
    )
- Workers
    Worker 1 → A.pdf
    Worker 2 → B.pdf
    Worker 3 → C.pdf

- Result:
    ✔ Parallel processing
    ✔ Faster than sequential
    ✔ Scales with CPU cores


------------------------------------------------------------------------------------------------------------------------------


FastAPI Event Loop (Brain)
   |
   |-- accepts requests
   |-- schedules coroutines
   |
   v
ThreadPoolExecutor (Workers)
   |
   |-- parse PDFs
   |-- extract images
   |-- run ML
   |
   v
Results back to Event Loop


------------------------------------------------------------------------------------------------------------------------------


- def ingest_files_sync(...):
    asyncio.run(self._ingest_sync(...))

- Why this exists
    - run_in_threadpool() expects a sync function
    - Your ingestion logic is async
    - You need a bridge

ThreadPool Worker
   ↓
Start new event loop
   ↓
Run async ingestion
   ↓
Close loop


































"""






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

"""
Data Ingestion Workflow:

: Step 0 – User uploads files (FastAPI)

    - goes to /upload file where { uploadFiles } function is called
    - Chat Repository initialized - { chat_repo = ChatRepository() ,inside db.chat_repository }
    - { chat_repo.create_session -> is Called } = Which returns a specific DB Session and corresponding session_id = DB session row created with your nice readable session_id (generate_session_id).


    - { ingestor = DataIngestor(session_id=session_id) } , which creates :
    - DataIngestor is constructed with the same session_id → this controls and creates:

        temp dir: data/{session_id}/
        faiss dir: faiss_index/{session_id}/
        artifacts dir: artifacts/{session_id}


    - After that in "/upload" route and {uploadFiles - function}:
    - Input files are wrapped as :
        wrapped = [FastAPIFileAdapter(f) for f in files]


    - Then { run_sync(ingestor.built_retriever, wrapped, "mmr", 20, 0.5) } is called
    - Inside { built_retriever } Function :

        Step 1 – Saving uploaded files

            { save_uploaded_files(uploaded_files, self.temp_dir) } Function is called which returns the List of Paths and following takes place:

            For each uploaded file:

                Clean, safe filename:
                original.pdf → original_abc12.pdf

                Saved into:
                data/{session_id}/original_abc12.pdf

                Metadata not touched yet (we’re still at file level).


        Step 2 – Converting files → Document objects

            { load_documents_and_assets(paths, images_dir, tables_dir) } Function is called which : Internally calls _process_single_path for each saved file in parallel (using executor).

            For each file type:

                - PDFs :
                    loader = PyPDFLoader(str(p))
                    text_docs = loader.load()
                    for doc in text_docs:
                        doc.metadata.update({
                            "modality": "text",
                            "source": str(p),  # full path to saved pdf
                        })


                - Then: Tables extracted via Camelot:

                    For each table → CSV + JSON saved under tables_dir
                    A table document is created:
                        Document(
                            page_content=table_text,         # small textual preview
                            metadata={
                                "modality": "table",
                                "source": str(p),
                                "saved_table_csv": str(csv_path),
                                "saved_table_json": str(json_path),
                                "page": t["page"],
                            }
                        )


                 - Images extracted with fitz:

                    Raw image bytes saved into images_dir
                    Caption is generated via Groq vision
                    One image document per caption:

                        Document(
                            page_content=caption,
                            metadata={
                                "modality": "image",
                                "source": str(p),
                                "page": page_number,
                            }
                        )


            So for PDFs, you get a mix of:

                modality="text"
                modality="table"
                modality="image"
            Each with "source" pointing to the actual file path.

            At the end of load_documents_and_assets, you have a flat List[Document] with rich metadata

            
        Step 3 – Multimodal chunking

            { DataIngestor._multimodal_split(docs, ...) } is Called:

            Creates:

                text splitter (big chunk size)
                table splitter (smaller chunk size)

            For each document:

                - If modality == "image" → no splitting, just carried on.

                - If modality == "table" → split via table splitter, create one Document per chunk with:

                    piece.metadata = dict(doc.metadata)
                    piece.metadata["modality"] = "table"

                - Else (text) → split_documents, then:

                    p.metadata.update(doc.metadata)
                    p.metadata["modality"] = "text"

            So now:

                - You have chunks: List[Document].
                - Each chunk still has modality, source, and possibly page, table paths, etc.








































"""
