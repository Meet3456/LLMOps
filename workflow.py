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