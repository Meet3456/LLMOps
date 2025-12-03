import asyncio
from concurrent.futures import ThreadPoolExecutor

IO_POOL_VAL = ThreadPoolExecutor(max_workers=8)


def run_sync(func, *args, **kwargs):
    """
    Run blocking / CPU-heavy / IO-heavy code off the current event loop.
    This is crucial for:
    - FAISS retrieval
    - Reranker / CrossEncoder
    - Groq LLM calls (LangChain)
    - File/embedding operations
    """
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(IO_POOL_VAL, lambda: func(*args, **kwargs))
