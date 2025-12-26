# backend/app/routes/documents.py

"""
Document management routes: upload, retrieval, and processing.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Request
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.dependencies.auth import get_current_user, get_current_user_or_dev_user  # DEV MODE ONLY
from app.models.user import User
from app.models.document import Document
from app.schemas.document import DocumentUploadFullResponse
from app.services.document_service import (
    save_upload_file,
    create_document_record,
    create_user_folder,
    read_raw_text,
    save_cleaned_text
)
from app.services.extractor import extract_text_auto
from app.utils.text_cleaner import clean_text
from app.services.retriever import index_document, is_document_indexed, get_index_stats
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Supported file extensions
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@router.post("/upload", response_model=DocumentUploadFullResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("20/hour")  # 20 uploads per hour
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user_or_dev_user),  # DEV MODE ONLY - Change back to get_current_user for production
    db: Session = Depends(get_db)
):
    """
    Upload a document, extract raw text, and clean text.
    
    Supported formats: PDF, DOCX, TXT
    
    Workflow:
    1. Validate file type
    2. Create user/document directory structure
    3. Save uploaded file
    4. Extract raw text
    5. Save raw.txt
    6. Clean text using NLP pipeline
    7. Save cleaned.txt
    8. Create database record
    9. Return document metadata with word counts
    
    Args:
        file: Uploaded file
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Document metadata with doc_id, filename, raw_word_count, clean_word_count
        
    Raises:
        HTTPException: 400 if file type is unsupported or text extraction fails
    """
    # Validate file extension
    filename = file.filename
    file_ext = None
    if filename:
        file_ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else None
    
    if not file_ext or file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Create a temporary document record to get doc_id
    temp_doc = create_document_record(
        db=db,
        user_id=current_user.id,
        filename=filename,
        text_path=None  # Will be updated after extraction
    )
    doc_id = temp_doc.id
    
    # Create user-specific folder structure
    user_doc_dir = create_user_folder(current_user.id, doc_id)
    
    # Save uploaded file
    saved_file_path = await save_upload_file(file, user_doc_dir, filename)
    
    # Extract raw text
    raw_text = extract_text_auto(saved_file_path)
    
    if not raw_text:
        # Clean up and raise error
        db.delete(temp_doc)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to extract text from document"
        )
    
    # Save raw text to raw.txt
    raw_text_path = user_doc_dir / "raw.txt"
    raw_text_path.write_text(raw_text, encoding="utf-8")
    
    # Calculate raw word count
    raw_word_count = len(raw_text.split())
    
    # Clean text using NLP pipeline
    cleaned_text = clean_text(raw_text)
    
    if not cleaned_text:
        # Clean up and raise error
        db.delete(temp_doc)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cleaning resulted in empty content"
        )
    
    # Save cleaned text to cleaned.txt
    cleaned_text_path = save_cleaned_text(current_user.id, doc_id, cleaned_text)
    
    # Calculate clean word count
    clean_word_count = len(cleaned_text.split())
    
    # Update document record with text_path
    temp_doc.text_path = str(raw_text_path)
    db.commit()
    db.refresh(temp_doc)
    
    return DocumentUploadFullResponse(
        doc_id=doc_id,
        filename=filename,
        raw_word_count=raw_word_count,
        clean_word_count=clean_word_count,
        message="Upload + cleaning successful"
    )


@router.post("/embed/{doc_id}")
async def embed_document(
    doc_id: int,
    current_user: User = Depends(get_current_user_or_dev_user),  # DEV MODE ONLY - Change back to get_current_user for production
    db: Session = Depends(get_db)
):
    """
    Create FAISS index for a document (EXPLICIT INDEXING STEP).
    
    This is a deliberate, separate step from upload/summarization.
    Must be called before using POST /ask/{doc_id}.
    
    Workflow:
    1. Validate document exists and user owns it
    2. Check if already indexed (idempotent)
    3. If not indexed: generate embeddings + build FAISS index
    4. Return index metadata
    
    Args:
        doc_id: Document ID to index
        current_user: Authenticated user
        db: Database session
        
    Returns:
        Index status and metadata
        
    Raises:
        404: Document not found or not owned by user
        400: Document has no cleaned text
        500: Indexing failed
        
    Example Response:
        {
            "status": "indexed",
            "doc_id": 42,
            "chunks": 12,
            "embedding_model": "all-mpnet-base-v2",
            "dimension": 768,
            "message": "Document indexed successfully"
        }
    """
    # Step 1: Validate document exists and user owns it
    document = db.query(Document).filter(
        Document.id == doc_id,
        Document.user_id == current_user.id
    ).first()
    
    if not document:
        print(f"⚠️ Embed failed: Document {doc_id} not found for user {current_user.id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {doc_id} not found or you don't have access"
        )
    
    # Step 2: Check if already indexed (idempotent operation)
    if is_document_indexed(current_user.id, doc_id):
        stats = get_index_stats(current_user.id, doc_id)
        print(f"ℹ️ Document {doc_id} already indexed ({stats['num_vectors']} chunks)")
        return {
            "status": "already_indexed",
            "doc_id": doc_id,
            "chunks": stats['num_vectors'],
            "embedding_model": "all-mpnet-base-v2",
            "dimension": stats['dimension'],
            "message": "Document is already indexed"
        }
    
    # Step 3: Verify cleaned text exists
    from pathlib import Path
    from app.core.config import get_settings
    settings = get_settings()
    
    cleaned_path = Path(settings.UPLOAD_DIR) / str(current_user.id) / str(doc_id) / "cleaned.txt"
    if not cleaned_path.exists():
        print(f"⚠️ Embed failed: No cleaned text for document {doc_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document {doc_id} has no cleaned text. Please upload it first."
        )
    
    # Step 4: Index the document (create embeddings + FAISS index)
    try:
        print(f"🔧 Starting indexing for document {doc_id} (user {current_user.id})...")
        index_document(current_user.id, doc_id)
        
        # Get stats after indexing
        stats = get_index_stats(current_user.id, doc_id)
        print(f"✅ Document {doc_id} indexed successfully ({stats['num_vectors']} chunks)")
        
        return {
            "status": "indexed",
            "doc_id": doc_id,
            "chunks": stats['num_vectors'],
            "embedding_model": "all-mpnet-base-v2",
            "dimension": stats['dimension'],
            "message": f"Document indexed successfully with {stats['num_vectors']} chunks"
        }
        
    except ValueError as e:
        print(f"❌ Indexing failed for document {doc_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to index document: {str(e)}"
        )
    except Exception as e:
        print(f"❌ Indexing failed for document {doc_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document: {str(e)}"
        )
