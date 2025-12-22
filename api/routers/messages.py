from fastapi import APIRouter, Depends, HTTPException

from db.chat_repository import ChatRepository
from db.database import get_db

router = APIRouter()


@router.get("/messages/{session_id}")
async def get_messages(session_id: str, db=Depends(get_db)):
    repo = ChatRepository()

    if not await repo.if_session_exists(db, session_id):
        raise HTTPException(404, "Session not found")

    messages = await repo.get_history(db, session_id, limit=1000)

    return [
        {"role": m.role, "content": m.content, "created_at": str(m.created_at)}
        for m in messages
    ]
