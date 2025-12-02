from fastapi import FastAPI, Depends, Header, HTTPException

# It is a great utility that reads environment variables and casts them to correct type:
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_key = str

    class Config:
        env_file = ".env"

settings = Settings()

app = FastAPI()
 
API_KEY = "my-secret-key"

def get_api_key_env_file(api_key: str = Header(...)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="unauthorized")
    else:
        return api_key


# (...) - This indicated the field is required
def get_api_key(api_key:str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="unauthorized")
    else:
        return api_key
    
@app.get('/get-data')

def get_data(api_key: str = Depends(get_api_key_env_file)):
    # Creating a dependency between the route handler and the get_api_key Function
    # if api_key is valid the function will return successfull response
    return {"output":"Access Granted"}



'''
- FastAPI is already fully concurrent - By keeping the routes async 
    [async def upload() ; async def chat()]

- As Data Ingestion for Faiss Creation and RAG Operations are CPU Heavy-  Not running them in event loop
  Thats why we use:
    [await run_sync(ingestor.built_retriver, ...) ; await run_sync(orchestrator.run_rag, ...)]
  Each user’s ingestion and chatting run on threadpool workers.

- IO_POOL = ThreadPoolExecutor(max_workers=32)
  This means:
    Up to 32 ingestion jobs can run at the same time.
    Up to 32 RAG/Groq calls can run at the same time.
    FAISS search + reranker + embeddings → offloaded from the event loop.

- index_path = f"faiss_index/{session_id}"
  Meaning:
    Each active user has their own FAISS index.

- if session_id not in self.cache:
    self.cache[session_id] = Orchestrator(index_path=f"faiss_index/{session_id}")

    This means:
        Each user gets a fully isolated orchestrator.

    Each orchestrator has:
        Its own retriever
        Its own FAISS instance
        Its own reranker
        Its own config
        Multiple orchestrators run concurrently → No shared state → No blocking.

    ➤ 5 users = 5 orchestrators = perfect concurrency.

- BenchMarking API's:
    - Latency
    - Throughput
    - Concurrency Handling
    - Error Rates
    - Resource Usage
'''