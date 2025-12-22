from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from db.chat_repository import ChatRepository
from db.database import get_db
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.src.document_ingestion.data_ingestion import DataIngestor

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
async def uploadFiles(
    files: list[UploadFile] = File(...),
    session_id: str | None = Form(None),
    db=Depends(get_db),
):
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
    if session_id:
        if not await chat_repo.if_session_exists(db,session_id):
            raise HTTPException(400, "Invalid session_id")
        log.info("Uploading to existing session | session_id=%s", session_id)
    else:
        session_id = chat_repo.create_session(db)
        log.info("Uploading created new session | session_id=%s", session_id)

    await chat_repo.set_ingestion_status(db, session_id, "indexing")

    try:
        # Building the Faiss index:
        ingestor = DataIngestor(session_id=session_id)

        # wrapping the files in FastAPIFileAdapter
        wrapped = [FastAPIFileAdapter(f) for f in files]

        # Run ingestion in threadpool (blocking work off main event loop)
        await ingestor.built_retriever(
            wrapped,
            chunk_size=2000,
            chunk_overlap=400,
            k=5,
            search_type="mmr",
            fetch_k=35,
            lambda_mult=0.5,
        )

        await chat_repo.add_files(
            db,
            session_id,
            [f.filename for f in files if f.filename],
        )

        await chat_repo.set_ingestion_status(db, session_id, "done")

        log.info(
            f"Upload completed and FAISS index built | session_id = {session_id} | files = {len(files)}"
        )
        return {"session_id": session_id, "indexed": True}
    except Exception as e:
        await chat_repo.set_ingestion_status(db, session_id, "failed")
        log.error("Upload failed | session_id=%s | error=%s", session_id, str(e))
        raise HTTPException(500, "Ingestion failed")
