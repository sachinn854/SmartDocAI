# backend/app/routes/summarize.py

"""
API routes for document summarization.

Design principles:
- Thin API layer (business logic in service)
- JWT protection (only authenticated users)
- Owner verification (users can only summarize their own documents)
- Caching (reuse existing summaries)
- Clear error messages
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
import logging
import gc

from app.core.database import get_db
from app.core.config import get_settings
from app.dependencies.auth import get_current_user, get_current_user_or_dev_user  # DEV MODE ONLY
from app.models.user import User
from app.models.document import Document
from app.schemas.document import DocumentSummaryResponse
from app.services.summarizer import (
    generate_all_summaries,
    save_summaries,
    load_summaries
)
from app.services.embeddings import unload_model as unload_embedding_model, load_embedding_model
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter(tags=["Summarization"])
settings = get_settings()
logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)


@router.post("/{doc_id}", response_model=DocumentSummaryResponse)
@limiter.limit("30/hour")  # 30 summarizations per hour
def summarize_document(
    request: Request,
    doc_id: int,
    current_user: User = Depends(get_current_user_or_dev_user),  # DEV MODE ONLY - Change back to get_current_user for production
    db: Session = Depends(get_db)
):
    """
    Generate AI summaries for a document.
    
    Generates three types of summaries:
    1. Short summary (30-40 words) - Quick overview
    2. Medium summary (150-200 words) - Comprehensive overview
    3. Detailed summary (bullet points) - Structured key insights
    
    Performance optimization:
    - Summaries are cached in summaries.json
    - If summaries already exist, they are returned immediately
    - Model is loaded once and reused across requests
    
    Workflow:
    1. Verify document exists and user owns it
    2. Check if summaries already cached
    3. If not, load cleaned.txt
    4. Generate summaries using T5
    5. Save to summaries.json
    6. Return summaries
    
    Args:
        doc_id: Document ID to summarize
        current_user: Authenticated user (JWT)
        db: Database session
        
    Returns:
        DocumentSummaryResponse with all three summary types
        
    Raises:
        404: Document not found or user doesn't own it
        404: cleaned.txt not found (run upload first)
        400: cleaned.txt is empty
        500: Model error during summarization
    """
    # Verify document exists
    document = db.query(Document).filter(Document.id == doc_id).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {doc_id} not found"
        )
    
    # Verify user owns the document
    if document.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID {doc_id} not found"  # Don't reveal existence
        )
    
    # Check if summaries already exist (caching)
    cached_summaries = load_summaries(current_user.id, doc_id)
    
    if cached_summaries:
        print(f"✅ Returning cached summaries for doc {doc_id}")
        return DocumentSummaryResponse(**cached_summaries)
    
    # Load cleaned text
    cleaned_text_path = settings.UPLOAD_DIR / str(current_user.id) / str(doc_id) / "cleaned.txt"
    
    if not cleaned_text_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="cleaned.txt not found. Please upload document first."
        )
    
    try:
        cleaned_text = cleaned_text_path.read_text(encoding='utf-8')
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading cleaned.txt: {str(e)}"
        )
    
    # Validate cleaned text is not empty
    if not cleaned_text or not cleaned_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="cleaned.txt is empty. Cannot generate summaries."
        )
    
    # ===== CRITICAL: Memory Management for Railway (512MB limit) =====
    # Unload embedding model before loading summarizer to prevent OOM
    if settings.ENV == "production":
        logger.info("🧹 [MEMORY] Unloading embedding model before summarization...")
        try:
            unload_embedding_model()
            gc.collect()  # Force garbage collection
            logger.info("✅ [MEMORY] Embedding model unloaded, memory freed")
        except Exception as e:
            logger.warning(f"⚠️  [MEMORY] Failed to unload embedding: {e}")
    
    # Generate summaries
    try:
        logger.info(f"🔄 Generating summaries for doc {doc_id} (user {current_user.id})")
        summaries = generate_all_summaries(cleaned_text)
        
        # Save summaries to disk (caching)
        save_summaries(current_user.id, doc_id, summaries)
        
        logger.info(f"✅ Summaries generated and cached for doc {doc_id}")
        
        return DocumentSummaryResponse(**summaries)
    
    except Exception as e:
        logger.error(f"❌ Error generating summaries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating summaries: {str(e)}"
        )
    
    finally:
        # ===== CRITICAL: Reload embedding model for future QA queries =====
        if settings.ENV == "production":
            logger.info("🔄 [MEMORY] Reloading embedding model after summarization...")
            try:
                load_embedding_model()  # Pre-warm for next QA request
                logger.info("✅ [MEMORY] Embedding model reloaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  [MEMORY] Failed to reload embedding: {e} (will load on demand)")
