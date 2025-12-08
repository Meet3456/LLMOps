from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from db.chat_repository import ChatRepository
from db.database import get_db
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.thread_pool import run_sync
from orchestrator.orchestrator_manager import orchestrator_manager
from redis_cache.redis_client import (
    cache_answer,
    cache_retrieval,
    get_cached_answer,
    get_cached_retrieval,
)

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str


@router.post("/chat")
async def chat(req: ChatRequest, db=Depends(get_db)):
    repo = ChatRepository()

    if not await repo.if_session_exists(db, req.session_id):
        raise HTTPException(400, "Invalid Session")

    query = req.message

    cached = get_cached_answer(query)

    if cached:
        return ChatResponse(answer=cached)

    orchestrator = orchestrator_manager.get_orchestrator(req.session_id)

    # 2. Retrieval cache
    doc_ids = get_cached_retrieval(req.session_id, query)
    if doc_ids:
        docs = orchestrator.retriever.return_docs_from_ids(doc_ids)
    else:
        docs = await run_sync(orchestrator.retriever.retrieve, query)

        ids = [d.metadata["doc_id"] for d in docs]
        log.info("List of retrieved doc ids : ", ids=ids)

        cache_retrieval(req.session_id, query, ids)

        # cache_retrieval(req.sesion_id , query , [d.id for d in docs])
