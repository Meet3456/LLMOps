from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from db.chat_repository import ChatRepository
from db.database import get_db
from multi_doc_chat.src.document_ingestion.data_ingestion import DataIngestor
from multi_doc_chat.utils.thread_pool import run_sync

router = APIRouter()


class FastAPIFileAdapter:
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename

    def getbuffer(self):
        self._uf.file.seek(0)
        return self._uf.file.read()


@router.post("/upload")
async def uploadFiles(files: list[UploadFile] = File(...), db=Depends(get_db)):
    if not files:
        raise HTTPException(400, "No files uploaded")

    chat_repo = ChatRepository()

    # creating session in the Datbase:
    session_id = await chat_repo.create_session(db=db)

    # Building the Faiss index:
    ingestor = DataIngestor(session_id=session_id)

    # wrapping the files in FastAPIFileAdapter
    wrapped = [FastAPIFileAdapter(f) for f in files]

    await run_sync(ingestor.built_retriever, wrapped, "mmr", 20, 0.5)

    return {"session_id": session_id, "indexed": True}