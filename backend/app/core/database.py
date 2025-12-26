# backend/app/core/database.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import get_settings

settings = get_settings()

# --------------------------------
# SQLAlchemy Base Class
# --------------------------------
class Base(DeclarativeBase):
    pass

# --------------------------------
# Engine
# --------------------------------
# For SQLite add check_same_thread=False
connect_args = {}

if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}
    # For HF Spaces, use single connection to avoid locking issues
    pool_size = 1
    max_overflow = 0
else:
    pool_size = 5
    max_overflow = 10

engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=pool_size,
    max_overflow=max_overflow,
    connect_args=connect_args
)

# --------------------------------
# SessionLocal
# --------------------------------
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# --------------------------------
# Dependency for FastAPI routes
# --------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
