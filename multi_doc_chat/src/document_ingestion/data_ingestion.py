from __future__ import annotations

import hashlib
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.document_ops import load_documents_and_assets
from multi_doc_chat.utils.file_io import save_uploaded_files
from multi_doc_chat.utils.model_loader import ModelLoader


# Function to generate a unique session ID: 
def generate_session_id() -> str:
    """Generate a unique session ID with timestamp."""
    now = datetime.now()

    day = now.strftime("%d")  # 18
    month = now.strftime("%b").lower()  # nov
    year = now.strftime("%Y")  # 2025
    time_part = now.strftime("%I:%M_%p")  # 03:13_PM

    # Clean time format (remove leading 0, lowercase am/pm)
    time_part = time_part.lstrip("0").lower()

    unique_id = uuid.uuid4().hex[:4]
    return f"session_{day}_{month}_{year}_{time_part}_{unique_id}"


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
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        try:
            # Object to load the necessary models
            self.model_loader = ModelLoader()

            # Use seeion based directories:
            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()

            # Initialize directories(temp,faiss and artifacts)
            self.temp_base = Path(temp_base)
            self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base)
            self.faiss_base.mkdir(parents=True, exist_ok=True)

            # underscore _ at the beginning (_resolve_dir) is a strong Python convention meaning this is an "internal" or "private" helper method, not meant to be called from outside the class.
            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            # New: artifact directory for saving extracted images/tables
            self.artifacts_base = Path("artifacts")
            self.artifacts_dir = self._resolve_dir(self.artifacts_base)

            # Subdirectories
            self.images_dir = self.artifacts_dir / "images"
            self.tables_dir = self.artifacts_dir / "tables"

            self.images_dir.mkdir(parents=True, exist_ok=True)
            self.tables_dir.mkdir(parents=True, exist_ok=True)

            log.info(
                "ChatIngestor initialized",
                session_id=self.session_id,
                temp_dir=str(self.temp_dir),
                faiss_dir=str(self.faiss_dir),
                sessionized=self.use_session,
                images=str(self.images_dir),
                tables=str(self.tables_dir),
            )

        except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException(
                "Initialization error in ChatIngestor", e
            ) from e

    def _resolve_dir(self, base_path: Path) -> Path:
        """Resolve directory path, optionally adding session ID."""
        # by default it is True , so it will create session specific directories
        if self.use_session:
            # e.g. set the dir_path to "faiss_index/abc123"
            dir_path = base_path / self.session_id
        else:
            # else fallback to as use_session is False: "faiss_index/"
            dir_path = base_path
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def _multimodal_split(
        self,
        docs: List[Document],
        chunk_size_text: int = 1000,
        chunk_overlap_text: int = 200,
        chunk_size_table: int = 600,
        chunk_overlap_table: int = 50,
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

        log.info(
            "Multimodal split complete",
            text_chunks=len(
                [c for c in out_chunks if c.metadata.get("modality", "text") == "text"]
            ),
            table_chunks=len(
                [c for c in out_chunks if c.metadata.get("modality") == "table"]
            ),
            image_chunks=len(
                [c for c in out_chunks if c.metadata.get("modality") == "image"]
            ),
            session_id=self.session_id,
        )
        return out_chunks

    async def built_retriever(
        self,
        paths: list[Path],
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
        search_type: str = "mmr",
        fetch_k: int = 35,
        lambda_mult: float = 0.5,
    ):
        log.info(
            "Starting ingestion: saving uploaded files",
            count=len(list(paths)),
            session_id=self.session_id,
        )
        # Step 1: persist files to temp dir (save_uploaded_files returns Path list)
        paths = save_uploaded_files(paths, self.temp_dir)
        log.info(
            "Files saved to temp dir",
            saved=[str(p) for p in paths],
            session_id=self.session_id,
        )

        # Step 2: async load docs & assets (text, tables, images/captions)
        docs = await load_documents_and_assets(
            paths, images_dir=self.images_dir, tables_dir=self.tables_dir
        )
        log.info(
            "Loaded documents & assets", count=len(docs), session_id=self.session_id
        )

        if not docs:
            raise ValueError("No valid documents loaded")

        # Step 3: chunking
        chunks = self._multimodal_split(
            docs,
            chunk_size_text=chunk_size,
            chunk_overlap_text=chunk_overlap,
            chunk_size_table=600,
            chunk_overlap_table=50,
        )
        log.info(
            "Total chunks after splitting",
            chunks=len(chunks),
            session_id=self.session_id,
        )

        # Step 3a: ensure each chunk has a stable unique ID in metadata
        for idx, c in enumerate(chunks):
            md = dict(c.metadata or {})
            # Only assign if not already present
            if "id" not in md:
                md["id"] = f"{self.session_id}__{idx}_{uuid.uuid4().hex[:8]}"
            c.metadata = md

        log.info(
            "Assigned stable IDs to chunks",
            total_chunks=len(chunks),
            example_id=chunks[0].metadata.get("id") if chunks else None,
            session_id=self.session_id,
        )

        # Step 4: create/load FAISS manager
        fm = FaissManager(self.faiss_dir, self.model_loader)

        texts = [c.page_content for c in chunks]
        metadatas = [dict(c.metadata or {}) for c in chunks]

        try:
            vs = fm.load_or_create_index(texts=texts, metadatas=metadatas)
            log.info(
                "FAISS loaded or created",
                index_dir=str(self.faiss_dir),
                session_id=self.session_id,
            )
        except Exception as e:
            log.warning(
                "First attempt to load/create FAISS failed, retrying",
                error=str(e),
                session_id=self.session_id,
            )
            vs = fm.load_or_create_index(texts=texts, metadatas=metadatas)

        # Step 5: add documents idempotently
        added = fm.add_documents(chunks)
        log.info("Added documnets to faiss",added = added)

        # Step 6: return retriever configured with search kwargs
        search_kwargs = {"k": k}

        if search_type == "mmr":
            search_kwargs.update({"fetch_k": fetch_k, "lambda_mult": lambda_mult})

        retriever = vs.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )
        log.info(
            "Retriever ready", search_type=search_type, k=k, session_id=self.session_id
        )
        return retriever


class FaissManager:
    """
    Manages a FAISS index directory with a small metadata file to avoid duplicate ingestion.
    - index_dir: directory where index.faiss and index.pkl are stored
    - ingested_meta.json: keeps track of already-ingested fingerprints
    """

    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}}

        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {
                    "rows": {}
                }
                log.info(
                    "Loaded existing FAISS metadata",
                    index_dir=str(self.index_dir),
                    entries=len(self._meta.get("rows", {})),
                )
            except Exception as e:
                self._meta = {"rows": {}}
                log.error(
                    "Failed to load FAISS metadata",
                    index_dir=str(self.index_dir),
                    error=str(e),
                )

        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None

    def _exists(self) -> bool:
        """
        This acts as the on-disk test to decide whether to load an index or create one.Returns True if both the FAISS index file and the associated metadata file exist in the specified index directory.
        """
        return (self.index_dir / "index.faiss").exists() and (
            self.index_dir / "index.pkl"
        ).exists()

    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        src = md.get("source", "unknown")
        return f"{src}::{h}"

    def _save_meta(self) -> None:
        self.meta_path.write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add_documents(self, docs: List[Document]):
        if self.vs is None:
            raise ValueError(
                "FAISS vectorstore not loaded. Call load_or_create_index() first."
            )

        new_docs: List[Document] = []

        for doc in docs:
            key = self._fingerprint(doc.page_content, doc.metadata or {})
            if key in self._meta.get("rows", {}):
                log.debug("Skipping already-ingested document", fingerprint=key)
                continue

            # store minimal data and diagnostics in metadata
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

                doc.metadata = md

            ids = [doc.metadata["id"] for doc in new_docs]
            self.vs.add_documents(new_docs, ids=ids)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()

            log.info(
                "Added new documents to FAISS index",
                new_count=len(new_docs),
                index_dir=str(self.index_dir),
            )

        return new_docs

    def load_or_create_index(
        self, texts: Optional[List[str]], metadatas: Optional[list[dict]]
    ):
        if self._exists():
            log.info("Loading existing FAISS index", index_dir=str(self.index_dir))

            self.vs = FAISS.load_local(
                str(self.index_dir), self.emb, allow_dangerous_deserialization=True
            )
            # Return loaded vectorstore
            return self.vs

        # Create new index if texts provided
        if not texts:
            raise DocumentPortalException(
                "No existing FAISS index and no data to create one", sys
            )

        # Ensure the id's exist in metadata:
        metadatas = metadatas or []

        for i, individual_md in enumerate(metadatas):
            if "id" not in individual_md:
                individual_md["id"] = f"doc_{i}_{uuid.uuid4().hex[:8]}"

        ids = [individual_md["id"] for individual_md in metadatas]

        self.vs = FAISS.from_texts(
            texts=texts, embedding=self.emb, metadatas=metadatas, ids=ids
        )
        self.vs.save_local(str(self.index_dir))
        self._save_meta()
        return self.vs
