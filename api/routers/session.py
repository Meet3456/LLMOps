import os
import shutil
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from db.chat_repository import ChatRepository
from db.database import get_db
from multi_doc_chat.logger import GLOBAL_LOGGER as log

router = APIRouter()


class SessionInfo(BaseModel):
    id: str
    created_at: str
    ingestion_status: str | None = None


@router.get("/sessions", response_model=List[SessionInfo])
async def list_sessions(db=Depends(get_db)):
    """
    List all sessions. Moved here from chat.py for better architecture.
    """
    repo = ChatRepository()
    sessions = await repo.list_sessions(db)

    return [
        {
            "id": s.id,
            "created_at": str(s.created_at),
            "ingestion_status": getattr(s, "ingestion_status", "unknown"),
        }
        for s in sessions
    ]


@router.post("/sessions")
async def create_session(db=Depends(get_db)):
    """
    Create a new session. Required by Streamlit 'New Chat' button.
    """
    repo = ChatRepository()
    session_id = await repo.create_session(db)
    log.info("Created new session | session_id=%s", session_id)
    return {"session_id": session_id}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db=Depends(get_db)):
    repo = ChatRepository()

    # Initially check if the given session exists corresponding to the session_id:
    if not await repo.if_session_exists(db, session_id):
        raise HTTPException(404, "Session not found")
    # call the delete session func:
    await repo.delete_session(db, session_id)

    # Remove FAISS index safely
    faiss_path = f"faiss_index/{session_id}"
    if os.path.exists(faiss_path):
        try:
            shutil.rmtree(faiss_path, ignore_errors=True)
            log.info("FAISS index removed | session_id=%s", session_id)
        except Exception as e:
            log.error(f"Error removing FAISS index: {e}")

    return {"deleted": True}
