import uuid

from sqlalchemy import TIMESTAMP, ForeignKey, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from multi_doc_chat.src.document_ingestion.data_ingestion import generate_session_id


class Base(DeclarativeBase):
    pass


class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: generate_session_id
    )
    created_at: Mapped[str] = mapped_column(TIMESTAMP, server_default=func.now())


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String, ForeignKey("sessions.id"))
    role: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(TIMESTAMP, server_default=func.now())
