from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.awsS3_client import S3Client
from multi_doc_chat.utils.document_ops import load_documents_and_assets
from multi_doc_chat.utils.file_io import save_uploaded_files
from multi_doc_chat.utils.model_loader import ModelLoader


class DataIngestor:
    """
    Ingest documents (text, pdf, images, tables) into a FAISS vectorstore.

    - save input files to temp_dir
    - extract text, tables, and image captions (via async loaders)
    - multimodal chunking (text / table / image-aware)
    - create or load FAISS index idempotently, add new chunks only
    - return a configured retriever (supports 'mmr' or 'similarity')
    """

    # As soon as the object of class is created , this will initialize temp and faiss directories for storing input data and faiss index session wise (data and faiss_index folder)
    def __init__(
        self,
        session_id: str,
    ):
        try:
            self.session_id = session_id

            self.temp_dir = Path("data") / session_id
            self.faiss_dir = Path("faiss_index") / session_id
            self.artifacts_dir = Path("artifacts") / session_id
            self.images_dir = self.artifacts_dir / "images"
            self.tables_dir = self.artifacts_dir / "tables"

            for p in [
                self.temp_dir,
                self.faiss_dir,
                self.images_dir,
                self.tables_dir,
            ]:
                p.mkdir(parents=True, exist_ok=True)

            self.model_loader = ModelLoader()
            self.s3_client = S3Client()
            log.info(f"DataIngestor Initialized | session={session_id}")

        except Exception as e:
            log.error(f"Failed to initialize ChatIngestor | error = {str(e)}")
            raise DocumentPortalException(
                "Initialization error in ChatIngestor", e
            ) from e

    # Sync wrapper (called from threadpool - as threadpool excepts a sync function )
    def ingest_files_blocking(
        self,
        uploaded_files,
        chunk_size,
        chunk_overlap,
        chunk_size_table,
        chunk_overlap_table,
    ):
        """
        This runs in a worker thread.
        It is blocking by design.
        """
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                self._ingest_async(
                    uploaded_files,
                    chunk_size,
                    chunk_overlap,
                    chunk_size_table,
                    chunk_overlap_table,
                )
            )
        finally:
            loop.close()

    # Actual async ingestion which runs on another event loop:
    async def _ingest_async(
        self,
        uploaded_files,
        chunk_size,
        chunk_overlap,
        chunk_size_table,
        chunk_overlap_table,
    ):
        try:
            # Save files
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            log.info(
                "Files saved | count=%d | session_id=%s", len(paths), self.session_id
            )

            # Load documents concurrently
            docs = await load_documents_and_assets(
                paths,
                images_dir=self.images_dir,
                tables_dir=self.tables_dir,
            )
            log.info("Documents loaded | count=%d", len(docs))

            if not docs:
                raise ValueError("No valid documents loaded")

            # Chunking
            chunks = self._multimodal_split(
                docs, chunk_size, chunk_overlap, chunk_size_table, chunk_overlap_table
            )

            for i, c in enumerate(chunks):
                c.metadata.setdefault(
                    "id", f"{self.session_id}_{i}_{uuid.uuid4().hex[:6]}"
                )

            # FAISS update
            fm = FaissManager(
                index_dir=self.faiss_dir,
                session_id=self.session_id,
                model_loader=self.model_loader,
            )
            fm.add_documents(chunks)
            log.info("Added documnets to faiss")

            # Upload artifacts
            self.s3_client.upload_directory(
                self.artifacts_dir, f"artifacts/{self.session_id}"
            )

            log.info("Ingestion completed | session=%s", self.session_id)

        except Exception as e:
            log.exception("Ingestion failed")
            raise DocumentPortalException("Ingestion failed", e)

    def _multimodal_split(
        self,
        docs: List[Document],
        chunk_size_text,
        chunk_overlap_text,
        chunk_size_table,
        chunk_overlap_table,
    ) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_text,
            chunk_overlap=chunk_overlap_text,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        )

        table_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_table,
            chunk_overlap=chunk_overlap_table,
            separators=["\n\n", "\n", " ", ""],
        )

        out_chunks: List[Document] = []

        for doc in docs:
            # get the modality of each doc as saved wrt to the convention inside {load_documents_and_assets} Function
            modality = doc.metadata.get("modality", "text")

            if modality == "image":
                doc.metadata = dict(doc.metadata or {})
                doc.metadata.setdefault("modality", "image")
                out_chunks.append(doc)

            elif modality == "table":
                parts = table_splitter.split_text(doc.page_content)
                for p in parts:
                    piece = Document(page_content=p, metadata=dict(doc.metadata or {}))
                    piece.metadata["modality"] = "table"
                    out_chunks.append(piece)

            else:
                parts = text_splitter.split_documents([doc])
                for p in parts:
                    p.metadata = dict(p.metadata or {})
                    p.metadata.update(doc.metadata or {})
                    p.metadata.setdefault("modality", "text")
                    out_chunks.append(p)

        log.info("Multimodal split complete")
        return out_chunks


class FaissManager:
    """
    Manages a FAISS index directory with a small metadata file to avoid duplicate ingestion.
    - index_dir: directory where index.faiss and index.pkl are stored
    - ingested_meta.json: keeps track of already-ingested fingerprints
    """

    def __init__(
        self,
        index_dir: Path,
        session_id: str,
        model_loader: Optional[ModelLoader] = None,
    ):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id

        # S3 Setup
        self.s3_client = S3Client()
        self.s3_prefix = f"faiss_index/{self.session_id}"

        # metadata of the docs
        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}}

        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None

        # Sync meta immediately on init to see what we have
        self._sync_from_s3()
        self._load_local_meta()

    def _sync_from_s3(self):
        """Download index files from S3 to Local if they don't exist."""
        if not (self.index_dir / "index.faiss").exists():
            log.info(
                f"Local index missing for {self.session_id}. Attempting S3 download..."
            )
            self.s3_client.download_directory(self.s3_prefix, self.index_dir)

    def _sync_to_s3(self):
        """Upload local index files to S3."""
        log.info(f"Syncing index to S3 | session_id={self.session_id}")
        self.s3_client.upload_directory(self.index_dir, self.s3_prefix)

    def _load_local_meta(self):
        """Load metadata from disk."""
        # if the metadta already exists in the respective metadata_path then load it into {_meta} variable
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {
                    "rows": {}
                }
            except Exception as e:
                log.error(f"Corrupt metadata, resetting. Error: {e}")
                self._meta = {"rows": {}}

    def _exists(self) -> bool:
        """
        This acts as the on-disk test to decide whether to load an index or create one.Returns True if both the FAISS index file and the associated metadata file exist in the specified index directory.
        """
        return (self.index_dir / "index.faiss").exists() and (
            self.index_dir / "index.pkl"
        ).exists()

    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        """
        Create a fingerprint hash for (text, source) pair to detect duplicates.
        """
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        src = md.get("source", "unknown")
        return f"{src}::{h}"

    def _save_meta(self) -> None:
        self.meta_path.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add_documents(self, docs: List[Document]):
        """
        Add new non-duplicate documents to FAISS.
        Duplicate detection is based on _fingerprint(text, metadata).
        """
        if self.vs is None:
            self.load_or_create_index()

        new_docs: List[Document] = []

        # check if some doc already exists in the FAISS Index via fingerprint created for each chunk/docs
        for doc in docs:
            # Create a fingerprint key for the doc with help of page content and some metadata
            key = self._fingerprint(doc.page_content, doc.metadata or {})

            # if the key already exists in the meta-data rows - Then skip it and continue
            if key in self._meta.get("rows", {}):
                log.debug("Skipping already-ingested document | fingerprint=%s", key)
                continue

            # else store minimal data and diagnostics in metadata
            self._meta["rows"][key] = {
                "source": doc.metadata.get("source"),
                "modality": doc.metadata.get("modality"),
                "length": len(doc.page_content),
            }

            new_docs.append(doc)

        if new_docs:
            # Ensuring new documents have attached ids:
            for i, doc in enumerate(new_docs):
                md = dict(doc.metadata or {})

                if "id" not in md:
                    md["id"] = (
                        f"doc_add_{len(self._meta.get('rows', {})) + i}_{uuid.uuid4().hex[:8]}"
                    )
                # update the meta-data for the respective doc
                doc.metadata = md

            # get the ids for all the new_docs that needs to be added in the Faiss vectore-store
            ids = [doc.metadata["id"] for doc in new_docs]

            # add the documents
            self.vs.add_documents(new_docs, ids=ids)

            self.vs.save_local(str(self.index_dir))

            # save the updated meta-data
            self._save_meta()

            # CRITICAL: Push to S3 immediately
            self._sync_to_s3()

            log.info(
                "Added new documents to FAISS index | new_count=%d | index_dir=%s",
                len(new_docs),
                str(self.index_dir),
            )

        return new_docs

    def load_or_create_index(self):
        """
        Load existing FAISS index if present; otherwise create a new one using given texts.
        Ensures docstore keys = metadata['id'].
        """
        if self._exists():
            log.info("Loading existing FAISS index | index_dir=%s", str(self.index_dir))

            self.vs = FAISS.load_local(
                str(self.index_dir), self.emb, allow_dangerous_deserialization=True
            )
            # Return loaded vectorstore
            return self.vs

        # Create EMPTY FAISS index
        log.info(
            "Creating new FAISS index with dummy vector | index_dir=%s",
            str(self.index_dir),
        )
        # Createing a dummy document to initialize FAISS dimension
        dummy_doc = Document(
            page_content="__faiss_init__",
            metadata={"id": "__faiss_init__", "source": "system", "modality": "system"},
        )

        # Create index with dummy doc
        self.vs = FAISS.from_documents([dummy_doc], embedding=self.emb)
        self.vs.save_local(str(self.index_dir))
        self._save_meta()

        # Save index and metadata
        self._sync_to_s3()
        return self.vs
