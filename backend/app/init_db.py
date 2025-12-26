#!/usr/bin/env python3
"""
Database initialization script.
Creates all tables defined in SQLAlchemy models.
"""

from app.core.database import Base, engine
from app.models import user, document

def init_db():
    """Create all database tables."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully!")

if __name__ == "__main__":
    init_db()
