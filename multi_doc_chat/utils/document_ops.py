from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from fastapi import UploadFile

def load_documents_from_files(file_paths: Iterable[Path]) -> List[Document]:
    docs: List[Document] = []
    try:
        for p in file_paths:
            extension = p.suffix.lower()
            if extension == ".pdf":
                loader = PyPDFLoader(str(p))
                text_docs = loader.load()

                # mark the metadata
                for doc in text_docs:
                    doc.metadata = dict(doc.metadata or {})
                    doc.metadata.update({"modality":"text","source":str(p)})  





    except Exception as e:
        pass
