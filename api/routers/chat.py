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


@router.post("/chat", response_model=ChatResponse)
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
    session_id = req.session_id.strip()
    input_query = req.message.strip()

    if not session_id:
        raise HTTPException(400, "session_id required")
    if not input_query:
        raise HTTPException(400, "message required")

    log.info("Chat request received | session_id=%s", session_id)

    # Chat repository which contains helper functions related to database:
    repo = ChatRepository()

    # 1. Validate session if it exists in the Database
    if not await repo.if_session_exists(db, session_id):
        raise HTTPException(400, "Invalid Session")

    norm_query = _normalize_query(input_query)

    # 2. Get the answer from the redis cache(if present = fastest)
    cached_ans = get_cached_answer(session_id, norm_query=norm_query)
    if cached_ans:
        log.info("Answer cache HIT | session_id=%s", session_id)
        return ChatResponse(answer=cached_ans)

    # 3. Get orchestrator & retriever for this session
    orchestrator = orchestrator_manager.get_orchestrator(session_id)
    retriever = orchestrator.retriever

    # 4. Embed query once (used for semantic cache + retrieval)
    query_embedding = await run_sync(retriever.embed_query, norm_query)

    # 5. Lookup retrieval cache for a query using: semantic or exact norm query match in the redis database
    cache_entry = lookup_retrieval_entry(session_id, norm_query, query_embedding)

    docs = None

    # if cache entry is found then fetch the doc ids from the cache and from ids respective document
    if cache_entry:
        doc_ids = cache_entry["doc_ids"]
        docs = retriever.return_docs_from_ids(ids=doc_ids)
        log.info(
            f"Reused cached retrieval docs | session_id={session_id} | List_doc_ids={doc_ids} | count_of_docs_retrieved={len(docs)}"
        )

    # if cache entry is not found then do normal retrieval:
    else:
        docs = await run_sync(retriever.retrieve, norm_query)

        # get the doc ids for storing inside cache of respective session and query
        # ( redis_client.setex(entry_key, ttl, json.dumps(entry)) )
        doc_ids = [
            d.metadata["id"]
            for d in docs
            if d.metadata.get("id") and not d.metadata["id"].startswith("__")
        ]

        log.info(
            f"NORMAL RETRIEVER - List of doc_ids of document retrieved for the given query : doc_ids = {doc_ids}",
        )

        if doc_ids:
            store_retrieved_result_entry(
                session_id=session_id,
                norm_query=norm_query,
                embedding=query_embedding,
                doc_ids=doc_ids,
            )

            log.info(
                "Stored retrieval cache | session_id=%s | docs=%d",
                session_id,
                len(doc_ids),
            )

        else:
            log.info("No docs to cache for retrieval", session_id=req.session_id)

    # 6. Load chat history from DB and limit it to 4-5 messages
    messages = await repo.get_history(
        db=db, session_id=session_id, limit=5
    )

    # Build the langchain compatible chat History
    chat_history = [
        HumanMessage(m.content) if m.role == "user" else AIMessage(m.content)
        for m in messages
    ]

    # Graph Execution:
    state = {
        "input":input_query,
        "chat_history":chat_history,
        "orchestrator":orchestrator,
        "docs":docs,
        "steps":[]
    } 

    try:
        # Invoke the graph   
        result = await run_sync(orchestrator.graph.invoke, state)
        answer = result["output"]

    except Exception as e:
        log.error(
            "Chat execution failed | error=%s",str(e),
        )
        raise HTTPException(500, "internal_error")

    # Add the user and ai message to the db
    await repo.add_message_to_db(
        db=db,
        session_id=session_id,
        messages=[
            ("user", input_query),
            ("assistant", answer),
        ],
    )
    # Cache the Final answer along with the normalized user input query to the cache
    cache_answer(session_id, norm_query, answer)

    log.info("Chat completed | session_id=%s", session_id)
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
