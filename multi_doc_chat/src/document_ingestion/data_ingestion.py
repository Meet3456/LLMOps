from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException
import json
import uuid
from datetime import datetime
from multi_doc_chat.utils.file_io import save_uploaded_files
from multi_doc_chat.utils.document_ops import load_documents_and_assets
import hashlib
import sys
import asyncio

# Function to generate a unique session ID:
def generate_session_id() -> str:
    """Generate a unique session ID with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}"


class DataIngestor:
    # As soon as the object of class is created , this will initialize temp and faiss directories for storing input data and faiss index session wise (data and faiss_index folder)
    def __init__(
        self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        try:
            self.model_loader = ModelLoader()

            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()

            self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)

            # underscore _ at the beginning (_resolve_dir) is a strong Python convention meaning this is an "internal" or "private" helper method, not meant to be called from outside the class.
            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            log.info(
                "ChatIngestor initialized",
                    session_id=self.session_id,
                    temp_dir=str(self.temp_dir),
                    faiss_dir=str(self.faiss_dir),
                    sessionized=self.use_session
                )

        except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e
        
    
    def _resolve_dir(self, base_path: Path) -> Path:
        """Resolve directory path, optionally adding session ID."""
        # by default it is True , so it will create session specific directories
        if self.use_session:
            # e.g. "faiss_index/abc123"
            dir_path = base_path / self.session_id
        else:
            # fallback: "faiss_index/"
            dir_path = base_path
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    

    def _multimodal_split(
        docs: List[Document],
        chunk_size_text: int = 1000,
        chunk_overlap_text: int = 200,
        chunk_size_table: int = 600,
        chunk_overlap_table: int = 50
    ) -> List[Document]:
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_text,
            chunk_overlap=chunk_overlap_text,
            separators=["\n## ", "\n### ","\n\n", "\n", " ", ""]
        )
        
        table_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_table, 
            chunk_overlap=chunk_overlap_table,   
            separators=["\n\n", "\n", " ", ""]
        )

        out_chunks: List[Document] = []

        for doc in docs:
            modality = doc.metadata.get("modality", "text")

            if modality == "image":
                out_chunks.append(doc)
            elif modality == "table":
                parts = table_splitter.split_text(doc.page_content)
                for p in parts:
                    doc = Document(
                        page_content=p,
                        metadata=doc.metadata
                    )
                    out_chunks.append(doc)
            else:
                parts = text_splitter.split_documents([doc])
                for p in parts:
                    p.metadata = dict(p.metadata or {})
                    p.metadata.update(doc.metadata or {})
                    out_chunks.append(p)

        log.info("Multimodal split complete", 
            text_chunks=len([c for c in out_chunks if c.metadata.get("modality","text")=="text"]),
            table_chunks=len([c for c in out_chunks if c.metadata.get("modality")=="table"]),
            image_chunks=len([c for c in out_chunks if c.metadata.get("modality")=="image"])
        )
        return out_chunks
    

    async def built_retriever(self,
        paths: list[Path],
        *,
        chunk_size:int = 1000, 
        chunk_overlap:int = 200,
        k:int = 5,
        search_type: str = "mmr",
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ):
        # Save uploaded files to temp directory
        paths = save_uploaded_files(paths, self.temp_dir)

        # Load documents and assets
        docs = await load_documents_and_assets(paths)
        
        if not docs:
            raise ValueError("No valid documents loaded")

        chunks = self._multimodal_split(
            docs,
            chunk_size_text=chunk_size,
            chunk_overlap_text=chunk_overlap,
            chunk_size_table=600,
            chunk_overlap_table=50
        )

        # FAISS manager very very important class for the docchat
        fm = FaissManager(self.faiss_dir, self.model_loader)
