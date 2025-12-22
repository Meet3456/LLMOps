import uuid

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.src.document_ingestion.data_ingestion import generate_session_id

from .models import Message, Session, UploadedFile


class ChatRepository:
    """
    Repository providing CRUD operations for Session + Message models.
    """

    async def create_session(self, db: AsyncSession) -> str:
        # generate a unique current time stamp based session id:
        sid = generate_session_id()
        # create a instance of Session and bind it with id:
        s = Session(id=sid)
        # add the session to db
        db.add(s)
        # commit the change
        await db.commit()
        # refresh the db
        await db.refresh(s)
        log.info("New session created | session_id=%s", s.id)
        # return the session id
        return s.id

    async def delete_session(self, db: AsyncSession, session_id: str):
        # delete the session corresponding to Session id(check already done at the api router)
        await db.execute(delete(Session).where(Session.id == session_id))
        # commit the changes
        await db.commit()
        log.info("Session deleted | session_id=%s", session_id)

    async def if_session_exists(self, db: AsyncSession, session_id: str) -> bool:
        out = await db.execute(select(Session).where(Session.id == session_id))
        exists = out.scalar() is not None

        log.info(
            "Session existence check for id: | session_id=%s | exists=%s",
            session_id,
            exists,
        )
        return exists

    async def set_ingestion_status(
        self, db: AsyncSession, session_id: str, status: str
    ):
        s = await db.get(Session, session_id)
        if not s:
            return

        s.ingestion_status = status
        await db.commit()

        log.info(
            "Ingestion status updated | session_id=%s | status=%s",
            session_id,
            status,
        )

    async def add_files(self, db, session_id: str, filenames: list[str]):
        for f in filenames:
            db.add(
                UploadedFile(id=str(uuid.uuid4()), session_id=session_id, filename=f)
            )
        await db.commit()

        log.info(
            "Uploaded files registered | session_id=%s | count=%d",
            session_id,
            len(filenames),
        )

    async def list_files(self, db, session_id: str):
        q = await db.execute(
            select(UploadedFile).where(UploadedFile.session_id == session_id)
        )
        files = q.scalars().all()

        log.info(
            "Listed uploaded files | session_id=%s | count=%d",
            session_id,
            len(files),
        )
        return files

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
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        # restore chronological order
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
