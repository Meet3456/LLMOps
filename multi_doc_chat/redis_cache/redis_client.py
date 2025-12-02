import os
import json
import redis
from utils.hashing_for_redis import hash_str
from multi_doc_chat.logger import GLOBAL_LOGGER as log

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB", "0"))

redis_client = redis.Redis(
    host = REDIS_HOST,
    port = REDIS_PORT,
    db = REDIS_DB,
    decode_responses = True,
    socket_timeout = 0.1,
    socket_connect_timeout = 0.1
)


# RAG Retrieval cache:
def cache_retrieval(session_id , query, doc_ids):
    key = f"retriever:{session_id}:{hash_str(query)}"
    redis_client.setex(key , 24*3600 , json.dumps(doc_ids))

def get_cached_retrieval(session_id , query):
    key = f"retriever:{session_id}:{hash_str(query)}"
    cached_data = redis_client.get(key)
    if cached_data: 
        log.info("[CACHE HIT] returning data from cache")
        return json.loads(cached_data)
    else:
        log.info("Data not found in cache , returning none")
        return None
   
    
# Caching the Final Rag Answer:
def cache_answer(query: str, answer: str):
    key = f"rag:answer:{hash_str(query)}"
    redis_client.setex(key, 24 * 3600, answer)

def get_cached_answer(query: str):
    key = f"rag:answer:{hash_str(query)}"
    return redis_client.get(key)