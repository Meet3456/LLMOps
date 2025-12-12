from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from db.chat_repository import ChatRepository
from db.database import get_db
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.src.document_ingestion.data_ingestion import DataIngestor
from multi_doc_chat.utils.thread_pool import run_sync

router = APIRouter()


class FastAPIFileAdapter:
    """
    Adapter to wrap FastAPI UploadFile so that the ingestion
    code can operate over generic 'uploaded_files'.
    Provides .filename and .getbuffer() like a memory buffer.
    """

    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self):
        self._uf.file.seek(0)
        return self._uf.file.read()


@router.post("/upload")
async def uploadFiles(files: list[UploadFile] = File(...), db=Depends(get_db)):
    """
    Upload endpoint:
      - Creates a new DB session row
      - Runs ingestion pipeline for this session
      - Builds FAISS index under faiss_index/{session_id}
    """
    if not files:
        raise HTTPException(400, "No files uploaded")

    chat_repo = ChatRepository()

    # creating session in the Datbase:
    session_id = await chat_repo.create_session(db=db)
    log.info("Upload started", session_id=session_id, num_files=len(files))

    # Building the Faiss index:
    ingestor = DataIngestor(session_id=session_id)

    # wrapping the files in FastAPIFileAdapter
    wrapped = [FastAPIFileAdapter(f) for f in files]

    # Run ingestion in threadpool (blocking work off main event loop)
    await ingestor.built_retriever(
        uploaded_files=wrapped,
        chunk_size=1000,
        chunk_overlap=200,
        k=5,
        search_type="mmr",
        fetch_k=35,
        lambda_mult=0.5,
    )

    log.info("Upload completed and FAISS index built", session_id=session_id)
    return {"session_id": session_id, "indexed": True}
