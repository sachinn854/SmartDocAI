# backend/app/schemas/document.py

from pydantic import BaseModel, ConfigDict, Field, field_validator
from datetime import datetime
from typing import Optional
import re


class DocumentBase(BaseModel):
    """Base document schema with shared fields."""
    filename: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Document filename",
        examples=["document.pdf"]
    )
    
    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate filename format and security."""
        # Remove any path traversal attempts
        if '..' in v or '/' in v or '\\' in v:
            raise ValueError('Invalid filename: path traversal not allowed')
        
        # Check for valid file extension
        allowed_extensions = ['.pdf', '.docx', '.txt', '.png', '.jpg', '.jpeg']
        if not any(v.lower().endswith(ext) for ext in allowed_extensions):
            raise ValueError(
                f'Invalid file type. Allowed types: {", ".join(allowed_extensions)}'
            )
        
        return v


class DocumentRead(DocumentBase):
    """Schema for reading document data."""
    id: int
    user_id: int
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class DocumentMeta(DocumentRead):
    """Extended schema with processing artifacts metadata."""
    text_path: Optional[str] = None
    summary_path: Optional[str] = None
    index_path: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class DocumentUploadResponse(BaseModel):
    """Response schema for document upload."""
    doc_id: int
    filename: str
    word_count: int
    message: str


class DocumentUploadFullResponse(BaseModel):
    """Response schema for document upload with cleaning."""
    doc_id: int
    filename: str
    raw_word_count: int
    clean_word_count: int
    message: str


class DocumentSummaryResponse(BaseModel):
    """Response schema for document summarization."""
    short_summary: str
    medium_summary: str
    detailed_summary: list[str]
    
    model_config = ConfigDict(from_attributes=True)
