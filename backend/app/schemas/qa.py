# backend/app/schemas/qa.py

"""
Pydantic schemas for Question Answering (RAG) endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class QuestionRequest(BaseModel):
    """
    Request schema for asking questions about a document with enhanced validation.
    """
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language question about the document",
        examples=["What is the main topic of this document?"]
    )
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate question format and content."""
        v = v.strip()
        
        if len(v) < 3:
            raise ValueError('Question must be at least 3 characters long')
        
        if len(v) > 500:
            raise ValueError('Question must be less than 500 characters')
        
        # Check if question contains actual words
        if not any(c.isalnum() for c in v):
            raise ValueError('Question must contain alphanumeric characters')
        
        return v


class WebResult(BaseModel):
    """
    Schema for web search result.
    """
    title: str
    snippet: str
    url: str


class AnswerResponse(BaseModel):
    """
    Response schema for question answering.
    
    Contains the answer, source information, and optionally web search results.
    """
    answer: str = Field(
        ...,
        description="Generated answer to the question"
    )
    used_web: bool = Field(
        default=False,
        description="Whether web search fallback was used"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Sources used for answer generation (document, web, etc.)"
    )
    web_results: List[WebResult] = Field(
        default_factory=list,
        description="Web search results if web fallback was used"
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Retrieval confidence score (0-1)"
    )
