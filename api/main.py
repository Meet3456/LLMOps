from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import (
    chat,
    data_upload,
    files,
    health,
    messages,
    session,
)
from db.database import init_db
from multi_doc_chat.logger import GLOBAL_LOGGER as log


# Use lifespan instead of deprecated on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Application startup initiated")
    await init_db()
    yield
    log.info("Application shutdown")


app = FastAPI(title="Multi-Document RAG Backend", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router Registration
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(chat.router, tags=["chat"])
app.include_router(data_upload.router, tags=["upload"])
app.include_router(
    session.router, tags=["session"]
)  # Now handles GET /sessions and POST /session
app.include_router(files.router, tags=["files"])
app.include_router(messages.router, tags=["messages"])


@app.get("/")
async def root():
    return {"message": "Backend is running"}
