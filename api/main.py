# api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, data_upload, health
from db.database import init_db

app = FastAPI(title="Multi-Document RAG Backend", version="1.0")

# Allow requests from your Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, whitelist domains instead
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(chat.router, tags=["chat"])  # /chat , /sessions
app.include_router(data_upload.router, tags=["upload"])  # /upload


# Startup event -> initialize DB tables
@app.on_event("startup")
async def startup_event():
    await init_db()


@app.get("/")
async def root():
    return {"message": "Backend is running!"}
