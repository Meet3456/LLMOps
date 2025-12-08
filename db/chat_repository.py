from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Message, Session


class ChatRepository:
    async def create_session(self, db: AsyncSession) -> str:
        s = Session()
        db.add(s)
        await db.commit()
        await db.refresh()
        return s.id

    async def if_session_exists(self, db: AsyncSession, session_id: str) -> bool:
        out = await db.execute(select(Session).where(Session.id == session_id))
        if out.scalar() is not None:
            return True
        else:
            return False

    async def add_message_to_db(
        self, db: AsyncSession, session_id: str, role: str, content: str
    ):
        message = Message(session_id=session_id, role=role, content=content)
        db.add(message)
        await db.commit()

    async def get_history(self, db: AsyncSession, session_id: str):
        out = db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at)
        )
        return out.scalars().all()

    async def list_sessions(self, db: AsyncSession):
        q = await db.execute(select(Session).order_by(Session.created_at.desc()))
        return q.scalars().all()
