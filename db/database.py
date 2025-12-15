import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from db.models import Base
from multi_doc_chat.logger import GLOBAL_LOGGER as log

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://chatuser:pass@localhost/chatdb"
)

engine = create_async_engine(
    DATABASE_URL, echo=False, future=True
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
    autoflush=False,
    autocommit=False,
)


async def init_db() -> None:
    """
    Initialize database and create tables if they do not exist.
    Should be called once at startup.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Database initialized and tables created")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an AsyncSession per request,
    and ensures proper cleanup.
    """
    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        await db.close()
