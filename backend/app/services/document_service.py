# backend/app/services/document_service.py

"""
Document service: file handling and database operations.
"""

from pathlib import Path
from typing import Optional

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.document import Document

settings = get_settings()


def create_user_folder(user_id: int, doc_id: int) -> Path:
    """
    Create directory structure for user's document.
    
    Structure: data/uploads/<user_id>/<doc_id>/
    
    Args:
        user_id: User ID
        doc_id: Document ID
        
    Returns:
        Path object to the created directory
    """
    user_doc_dir = settings.UPLOAD_DIR / str(user_id) / str(doc_id)
    user_doc_dir.mkdir(parents=True, exist_ok=True)
    return user_doc_dir


async def save_upload_file(
    upload_file: UploadFile,
    destination_dir: Path,
    filename: str
) -> Path:
    """
    Save uploaded file to specified directory.
    
    Args:
        upload_file: FastAPI UploadFile object
        destination_dir: Directory to save file
        filename: Name to save file as
        
    Returns:
        Path to saved file
    """
    file_path = destination_dir / filename
    
    # Read and write file in chunks to handle large files
    with file_path.open("wb") as buffer:
        content = await upload_file.read()
        buffer.write(content)
    
    return file_path


def read_raw_text(user_id: int, doc_id: int) -> Optional[str]:
    """
    Read raw text from raw.txt file.
    
    Args:
        user_id: User ID
        doc_id: Document ID
        
    Returns:
        Raw text content or None if file doesn't exist
    """
    raw_text_path = settings.UPLOAD_DIR / str(user_id) / str(doc_id) / "raw.txt"
    
    if not raw_text_path.exists():
        return None
    
    try:
        return raw_text_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading raw text: {e}")
        return None


def save_cleaned_text(user_id: int, doc_id: int, cleaned_text: str) -> Path:
    """
    Save cleaned text to cleaned.txt file.
    
    Args:
        user_id: User ID
        doc_id: Document ID
        cleaned_text: Cleaned text content
        
    Returns:
        Path to saved cleaned.txt file
    """
    cleaned_text_path = settings.UPLOAD_DIR / str(user_id) / str(doc_id) / "cleaned.txt"
    
    # Write cleaned text with UTF-8 encoding
    cleaned_text_path.write_text(cleaned_text, encoding="utf-8")
    
    return cleaned_text_path


def create_document_record(
    db: Session,
    user_id: int,
    filename: str,
    text_path: Optional[str] = None
) -> Document:
    """
    Create a new document record in database.
    
    Args:
        db: Database session
        user_id: Owner user ID
        filename: Original filename
        text_path: Path to raw text file (optional)
        
    Returns:
        Created Document object
    """
    new_document = Document(
        user_id=user_id,
        filename=filename,
        text_path=text_path,
        summary_path=None,
        index_path=None
    )
    
    db.add(new_document)
    db.commit()
    db.refresh(new_document)
    
    return new_document
