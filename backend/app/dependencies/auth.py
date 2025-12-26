# backend/app/dependencies/auth.py

"""
Authentication dependencies for FastAPI routes.
"""

from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import decode_access_token
from app.core.config import get_settings
from app.models.user import User

# OAuth2 scheme for token extraction from Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# DEV MODE ONLY - Optional OAuth2 scheme for development bypass
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)



async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[Session, Depends(get_db)]
) -> User:
    """
    Extract and validate JWT token, then fetch the current user.
    
    Args:
        token: JWT token from Authorization: Bearer header
        db: Database session
        
    Returns:
        Current authenticated User object
        
    Raises:
        HTTPException: 401 if token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Decode token
    payload = decode_access_token(token)
    
    # Extract email from token payload
    email: str = payload.get("sub")
    if email is None:
        raise credentials_exception
    
    # Fetch user from database
    user = db.query(User).filter(User.email == email).first()
    
    if user is None:
        raise credentials_exception
    
    return user


# ============================================================
# DEV MODE ONLY - SAFE FOR PRODUCTION DEPLOYMENT
# ============================================================
async def get_current_user_or_dev_user(
    db: Annotated[Session, Depends(get_db)],
    token: Annotated[str | None, Depends(oauth2_scheme_optional)] = None
) -> User:
    """
    Environment-aware authentication dependency.
    
    Behavior:
    - If Authorization header present → validate JWT (all environments)
    - If missing AND ENV=development → return dummy dev user (local testing only)
    - If missing AND ENV=production → raise 401 error (secure)
    
    This is SAFE for production deployment because:
    1. Production environment automatically enforces authentication
    2. Dev mode only works when DATABASE_URL is SQLite (local)
    3. Railway automatically sets DATABASE_URL to PostgreSQL
    
    Args:
        db: Database session
        token: Optional JWT token from Authorization header
        
    Returns:
        Authenticated User (if token provided) or dev dummy user (dev only)
        
    Raises:
        HTTPException: 401 if token invalid or missing in production mode
    """
    import logging
    logger = logging.getLogger(__name__)
    settings = get_settings()
    
    # If token is provided, validate it (production behavior)
    if token:
        try:
            payload = decode_access_token(token)
            email: str = payload.get("sub")
            if email is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            user = db.query(User).filter(User.email == email).first()
            if user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            return user
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # No token provided - check environment
    if settings.ENV == "development":
        # DEV MODE ONLY - DO NOT COMMIT TO DB
        logger.warning(
            "⚠️  DEV MODE: Using dummy user (dev@localhost). "
            "This only works locally with SQLite!"
        )
        # Create ephemeral dev user (not persisted to database)
        dev_user = User(
            id=0,
            email="dev@localhost",
            hashed_password=""
        )
        # Manually set created_at to avoid DB default dependency
        from datetime import datetime, timezone
        dev_user.created_at = datetime.now(timezone.utc)
        
        return dev_user
    else:
        # Production mode - require authentication
        logger.error("Production mode: Authentication token required but not provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
# ============================================================
# END DEV MODE
# ============================================================
