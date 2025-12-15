from __future__ import annotations

import hashlib
import json
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
            # creates a session-wise artifacts folder based on session id(industry standard convention)
            self.artifacts_dir = self._resolve_dir(self.artifacts_base)

            # Subdirectories for images and tables
            self.images_dir = self.artifacts_dir / "images"
            self.tables_dir = self.artifacts_dir / "tables"

            self.images_dir.mkdir(parents=True, exist_ok=True)
            self.tables_dir.mkdir(parents=True, exist_ok=True)

            log.info(
                "ChatIngestor initialized",
            )

        except Exception as e:
            log.error(f"Failed to initialize ChatIngestor | error = {str(e)}")
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
            f"Starting ingestion: saving uploaded files | count = {len(list(paths))}"
        )

        # Step 1: persist files to temp dir (save_uploaded_files returns Path list)
        paths = save_uploaded_files(paths, self.temp_dir)
        log.info("Files saved | count=%d | session_id=%s", len(paths), self.session_id)

        # Step 2: async load docs & assets (text, tables, images/captions) from the list of paths
        docs = await load_documents_and_assets(
            paths, images_dir=self.images_dir, tables_dir=self.tables_dir
        )
        log.info("Documents loaded | count=%d", len(docs))

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

        # Step 3a: ensure each chunk has a stable unique ID in metadata
        for idx, individual_c in enumerate(chunks):
            md = dict(individual_c.metadata or {})
            # Only assign if not already present
            if "id" not in md:
                md["id"] = f"{self.session_id}__{idx}_{uuid.uuid4().hex[:8]}"
            individual_c.metadata = md

        log.info("Chunk IDs assigned | total_chunks=%d", len(chunks))

        # Step 4: create/load FAISS manager
        fm = FaissManager(self.faiss_dir, self.model_loader)

        # load or create faiss index
        try:
            vs = fm.load_or_create_index()
            log.info(
                f"FAISS loaded or created | index_dir = {str(self.faiss_dir)} | session_id = {self.session_id}"
            )

        except Exception as e:
            log.warning(
                f"First attempt to load/create FAISS failed, retrying | error={str(e)} | session_id = {self.session_id}"
            )
            vs = fm.load_or_create_index()

        # Step 5: add documents idempotently
        fm.add_documents(chunks)
        log.info("Added documnets to faiss")

        # Step 6: return retriever configured with search kwargs
        search_kwargs = {"k": k}

        if search_type == "mmr":
            search_kwargs.update({"fetch_k": fetch_k, "lambda_mult": lambda_mult})

        retriever = vs.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )
        log.info(
            f"Retriever ready | search_type = {search_type} | k = {k} | session_id = {self.session_id}"
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

        # metadata of the docs
        self._meta: Dict[str, Any] = {"rows": {}}

        # if the metadta already exist s in the respective metadata_path then load it into {_meta} variable
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {
                    "rows": {}
                }
                log.info(
                    "Loaded existing FAISS metadata | entries=%d | index_dir=%s",
                    len(self._meta.get("rows", {})),
                    str(self.index_dir),
                )
            # If it does not exists then initialize it as a wmpty dictionary with key - "rows"
            except Exception as e:
                self._meta = {"rows": {}}
                log.error(
                    "Failed to load FAISS metadata | error=%s | index_dir=%s",
                    str(e),
                    str(self.index_dir),
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
            raise ValueError(
                "FAISS vectorstore not loaded. Call load_or_create_index() first."
            )

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
            # save the updated Faiss index
            self.vs.save_local(str(self.index_dir))
            # save the updated meta-data
            self._save_meta()

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

        # Save index and metadata
        self.vs.save_local(str(self.index_dir))
        self._save_meta()

        return self.vs
