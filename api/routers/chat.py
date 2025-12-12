from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from db.chat_repository import ChatRepository
from db.database import get_db
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.thread_pool import run_sync
from orchestrator.orchestrator_manager import orchestrator_manager
from redis_cache.redis_client import (
    cache_answer,
    get_cached_answer,
    lookup_retrieval_entry,
    store_retrieved_result_entry,
)

router = APIRouter()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str


class SessionInfo(BaseModel):
    id: str
    created_at: str


def _normalize_query(q: str) -> str:
    """
    Normalize user query for better cache keys:
      - strip spaces , lowercase , collapse multiple spaces
    """
    return " ".join(q.lower().strip().split())


@router.post("/chat")
async def chat(req: ChatRequest, db=Depends(get_db)):
    """
    Main chat endpoint.

    Pipeline:
      1. Validate session
      2. Normalize query
      3. Answer cache lookup
      4. Retrieval cache lookup (exact + semantic)
      5. If cache miss → full retrieval
      6. Build chat history from DB
      7. Call Orchestrator.run_rag(...)
      8. Persist messages
      9. Cache answer
    """
    session_id = req.session_id
    log.info("Chat initiated for the follosing session : ", s_id=session_id)
    if not session_id:
        raise HTTPException(status_code=400, detail="no sepecific session")

    input_query = req.message

    if not input_query:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Chat repository which contains helper functions related to database:
    repo = ChatRepository()

    # 1. Validate session
    if not await repo.if_session_exists(db, session_id):
        raise HTTPException(400, "Invalid Session")

    norm_query = _normalize_query(input_query)

    log.info(
        "Chat request received",
        session_id=session_id,
        query=input_query,
        norm_query=norm_query,
    )

    # 2. Get the answer from the redis cache(if present = fastest)
    cached_ans = get_cached_answer(session_id, norm_query=norm_query)
    if cached_ans:
        log.info("Answer cache HIT", session_id=session_id)
        return ChatResponse(answer=cached_ans)

    # 3. Get orchestrator & retriever for this session
    orchestrator = orchestrator_manager.get_orchestrator(session_id)
    retriever = orchestrator.retriever

    # 4. Embed query once (used for semantic cache + retrieval)
    query_embedding = await run_sync(retriever.embed_query, norm_query)

    # 5. Lookup retrieval cache for a query using: semantic or exact norm query match in the redis database
    cache_entry = lookup_retrieval_entry(
        session_id, norm_query, query_embedding, semantic_threshold=0.9
    )

    # if cache entry is found then fetch the doc ids from the cache and from ids respective document
    if cache_entry:
        doc_ids = cache_entry["doc_ids"]
        docs = retriever.return_docs_from_ids(ids=doc_ids)
        log.info(
            "Reused cached retrieval docs",
            session_id=req.session_id,
            doc_ids=doc_ids,
            count=len(docs),
        )

    # if cache entry is not found then do normal retrieval:
    else:
        docs = await run_sync(retriever.retrieve, norm_query)

        # get the doc ids for storing inside cache of respective session and query
        # ( redis_client.setex(entry_key, ttl, json.dumps(entry)) )
        doc_ids = [d.metadata["id"] for d in docs if d.metadata.get("id") is not None]

        log.info(
            "List of doc if for document retrieved for the given query : ",
            list_ids=doc_ids,
        )

        if doc_ids:
            store_retrieved_result_entry(
                session_id=session_id,
                norm_query=norm_query,
                embedding=query_embedding,
                doc_ids=doc_ids,
            )

            log.info("Stored retrieval in cache", session_id=req.session_id)

        else:
            log.info("No docs to cache for retrieval", session_id=req.session_id)

    # 6. Load chat history from DB
    messages = await repo.get_history(db=db, session_id=session_id)
    chat_history = []

    for m in messages:
        if m.role == "user":
            chat_history.append(HumanMessage(content=m.content))
        else:
            chat_history.append(AIMessage(content=m.content))

    # 7. Run RAG pipeline through Orchestrator
    answer = await run_sync(orchestrator.run_rag, input_query, chat_history)

    # 8. Persist messages to DB
    await repo.add_message_to_db(db, req.session_id, "user", input_query)
    await repo.add_message_to_db(db, req.session_id, "assistant", answer)

    # 9. Cache final answer
    cache_answer(req.session_id, norm_query, answer)

    log.info("Chat turn completed", session_id=req.session_id)
    return ChatResponse(answer=answer)


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(db=Depends(get_db)):
    """
    List all sessions to power the "previous chats" list in the frontend.
    """
    repo = ChatRepository()
    sessions = await repo.list_sessions(db)

    out: list[SessionInfo] = []
    for s in sessions:
        out.append(SessionInfo(id=s.id, created_at=str(s.created_at)))
    return out
