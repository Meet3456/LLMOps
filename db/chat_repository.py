from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from multi_doc_chat.logger import GLOBAL_LOGGER as log

from .models import Message, Session


class ChatRepository:
    """
    Repository providing CRUD operations for Session + Message models.
    """

    async def create_session(self, db: AsyncSession) -> str:
        s = Session()
        db.add(s)
        await db.commit()
        await db.refresh(s)
        log.info("New session created | session_id=%s", s.id)
        return s.id

    async def if_session_exists(self, db: AsyncSession, session_id: str) -> bool:
        out = await db.execute(select(Session).where(Session.id == session_id))
        exists = out.scalar() is not None

        log.info(
            "Session existence check | session_id=%s | exists=%s",
            session_id,
            exists,
        )
        return exists

    async def add_message_to_db(
        self,
        db: AsyncSession,
        session_id: str,
        messages: list[tuple[str, str]],
    ):
        objs = [Message(session_id=session_id, role=r, content=c) for r, c in messages]
        db.add_all(objs)
        await db.commit()

        log.info(
            "Messages persisted | session_id=%s | count=%d",
            session_id,
            len(objs),
        )
    async def get_history(self, db: AsyncSession, session_id: str, limit: int):
        """
        Get all messages of a session ordered by creation time.
        Used to build chat_history for RAG.
        """
        out = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at)
            .limit(limit)
        )
        rows = list(reversed(out.scalars().all()))
        log.info(
            "Loaded recent history | session_id=%s | count=%d",
            session_id,
            len(rows),
        )
        return rows

    async def list_sessions(self, db: AsyncSession):
        """
        List all sessions sorted by most recent first.
        Used by frontend to show "previous chats" like ChatGPT.
        """
        q = await db.execute(select(Session).order_by(Session.created_at.desc()))
        sessions = q.scalars().all()
        log.info(f"Listing sessions | count = {len(sessions)}")
        return sessions
