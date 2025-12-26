# backend/app/models/document.py

from sqlalchemy import String, Integer, DateTime, ForeignKey, func
from sqlalchemy.orm import mapped_column, Mapped, relationship
from datetime import datetime
from typing import Optional

from app.core.database import Base


class Document(Base):
    """Document model for storing uploaded files and their processed artifacts."""
    
    __tablename__ = "documents"
    
    # Primary Key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    # Foreign Key to User
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # File Metadata
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Processed File Paths (nullable until processing completes)
    text_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    summary_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    index_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    
    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="documents")
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename={self.filename}, user_id={self.user_id})>"
