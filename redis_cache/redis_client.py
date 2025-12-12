import json
import math
import os
from typing import Dict, List, Optional

import redis

from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.hashing_for_redis import hash_str

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    decode_responses=True,
    socket_timeout=2.0,
    socket_connect_timeout=2.0,
)


def cache_answer(session_id: str, norm_query: str, answer: str, ttl: int = 86400):
    """
    Cache the final LLM Answer for (session_id,normalized_query)
    """
    key = f"ans:{session_id}:{hash_str(norm_query)}"
    try:
        redis_client.setex(key, ttl, answer)
        log.debug("Cached answer for key", key=key)
    except Exception as e:
        log.warning("Failed to cache answer", error=str(e))


def get_cached_answer(session_id: str, norm_query: str) -> Optional[str]:
    """
    Retrieve cached final answer for (session_id, normalized_query).
    """
    key = f"ans:{session_id}:{hash_str(norm_query)}"

    try:
        value = redis_client.get(key)
        if value:
            log.debug(
                "LLM Answer successfully retrieved from cache for key : ", key=key
            )
        else:
            log.debug("Answer cache MISS", key=key)
        return value
    except Exception as e:
        log.warning("Failed to fetch cached answer", error=str(e))
        return None


def _session_query_index_key(session_id: str):
    """
    Returns a Redis key that stores a SET of query hashes for a session.
    example : ret:index:session_18_nov_2025_3:13_pm_ab12

    """
    return f"ret:index:{session_id}"


def _session_query_entry_key(session_id: str, query_hash: str) -> str:
    """
    example : retq:session_18_nov_2025_3:13_pm_ab12<session_id>:f7e9a1b04e<query_hash>
    And this key stores the actual JSON payload:

    {
      "norm_query": "what is the dropout rate",
      "embedding": [0.12, 0.98, ...],
      "doc_ids": [
        "session_x__12_abcd89ef",
        "session_x__13_ffe093bc"
      ]
    }
    """
    return f"retq:{session_id}:{query_hash}"


def cosine_sim(v1: List[float], v2: List[float]):
    if not v1 or not v2 or len(v1) != len(v2):
        return -1.0

    dot = sum(a * b for a, b in zip(v1, v2))

    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 == 0 or norm2 == 0:
        return -1.0

    return dot / (norm1 * norm2)


def store_retrieved_result_entry(
    session_id,
    norm_query: str,
    embedding: List[float],
    doc_ids: List[str],
    ttl: int = 86400,
):
    """
    Store the retrieved result from the retrieval process for a query:
      - normalized query text
      - embedding vector
      - final doc_ids used for RAG
    Also adds query hash to per-session index (for later semantic lookup).
    """
    try:
        # hash the input query:
        q_hash = hash_str(norm_query)
        # store the hash query key by mapping it to respective session using (_session_query_entry_key)
        entry_key = _session_query_entry_key(session_id, q_hash)

        # create a var for storing the input query , its corresponding embedding and the retrieved docs
        entry = {
            "norm_query": norm_query,
            "embedding": embedding,
            "doc_ids": doc_ids,
        }

        # store the entry into redis
        redis_client.setex(entry_key, ttl, json.dumps(entry))

        # create a session index set:
        idx_key = _session_query_index_key(session_id)
        # add the respective query hash inside the session index set(idx_key)
        redis_client.sadd(idx_key, q_hash)
        # expiration
        redis_client.expire(idx_key, ttl)

        log.debug(
            "Stored retrieval entry",
            session_id=session_id,
            norm_query=norm_query,
            doc_ids_count=len(doc_ids),
        )
    except Exception as e:
        log.warning("Failed to store retrieval entry", error=str(e))


def lookup_retrieval_entry(
    session_id: str,
    norm_query: str,
    query_embedding: List[float],
    semantic_threshold: float = 0.9,
) -> Optional[Dict]:
    """
    Lookup retrieval cache for a query using:
      1. Exact normalized text key
      2. Semantic similarity against previous queries in the session

    Returns:
      entry dict {norm_query, embedding, doc_ids} if matched or None.
    """
    try:
        q_hash = hash_str(norm_query)
        exact_key = _session_query_entry_key(session_id, q_hash)

        # Case:1 Exact match - if user asks the similar question(exact wordings) again

        cached_docs = redis_client.get(exact_key)
        if cached_docs:
            log.debug("Retrieval cache HIT (exact)", session_id=session_id)
            return json.loads(cached_docs)

        # Case:2 - Semantic matching between the current user query and the cached queries:

        # get the set of respective session (iske andar sare prev hashed query honge)
        idx_key = _session_query_index_key(session_id)
        prev_hashed_query_keys = redis_client.smembers(idx_key)

        if not prev_hashed_query_keys:
            log.debug(
                "Retrieval cache MISS (no index for the respective session)",
                session_id=session_id,
            )
            return None

        best_entry = None
        best_sim = -1.0

        for hash in prev_hashed_query_keys:
            query_key = _session_query_entry_key(session_id, hash)
            raw_entry = redis_client.get(query_key)
            if not raw_entry:
                continue
            entry = json.loads(raw_entry)
            embd = entry.get("embedding", [])
            similarity = cosine_sim(query_embedding, embd)
            if similarity > best_sim:
                best_sim = similarity
                best_entry = entry

        if best_entry and best_sim >= semantic_threshold:
            log.debug(
                "Retrieval cache HIT (semantic)",
                session_id=session_id,
                sim=best_sim,
            )
            return best_entry

        log.debug(
            "Retrieval cache MISS (semantic)", session_id=session_id, best_sim=best_sim
        )
        return None

    except Exception as e:
        log.warning("Failed to lookup retrieval entry", error=str(e))
        return None
