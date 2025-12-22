from fastapi import APIRouter, Depends, HTTPException

from db.chat_repository import ChatRepository
from db.database import get_db

router = APIRouter()


@router.get("/files/{session_id}")
async def list_files(session_id: str, db=Depends(get_db)):
    repo = ChatRepository()

    if not await repo.if_session_exists(db, session_id):
        raise HTTPException(404, "Session not found")

    files = await repo.list_files(db, session_id)
    return [{"filename": f.filename, "created_at": str(f.created_at)} for f in files]
