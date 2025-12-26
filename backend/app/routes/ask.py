# backend/app/routes/ask.py

"""
 Question Answering API Endpoint (Extractive QA).

Provides POST /ask/{doc_id} endpoint with:
- Document-first extractive QA pipeline
- Sentence-level extraction
- Optional LLM simplification
- Web search fallback (only when document has no answer)
- Transparent responses
- Fail-fast validations
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.dependencies.auth import get_current_user, get_current_user_or_dev_user  # DEV MODE ONLY
from app.models.user import User
from app.services.embeddings import embed_query, generate_document_embeddings
from app.services.retriever import is_document_indexed, retrieve_similar_chunks, load_faiss_index
from app.services.qa import (
    extract_answer_from_document, 
    simplify_answer_with_llm,
    compute_extractive_confidence,
    generate_answer_from_context,  # For web fallback only
    compute_confidence  # For web fallback only
)
from app.services.websearch import get_web_context_for_question
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

settings = get_settings()
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Confidence threshold for web fallback
CONFIDENCE_THRESHOLD = 0.70

# Number of chunks to retrieve
TOP_K_CHUNKS = 5


def deduplicate_chunks(chunks: List[str], scores: List[float], similarity_threshold: float = 0.85) -> tuple[List[str], List[float]]:
    """
    Remove near-duplicate chunks based on text similarity.
    
    Keeps only distinct chunks to prevent repetitive context
    and improve answer quality.
    
    Args:
        chunks: List of text chunks from retrieval
        scores: Corresponding similarity scores
        similarity_threshold: Threshold for considering chunks as duplicates (0-1)
        
    Returns:
        Tuple of (deduplicated_chunks, deduplicated_scores)
        
    Notes:
        - Uses simple character overlap ratio for speed
        - Preserves order (highest scoring chunks first)
        - Always keeps first (highest scoring) chunk
    """
    if not chunks:
        return [], []
    
    unique_chunks = []
    unique_scores = []
    
    for i, chunk in enumerate(chunks):
        is_duplicate = False
        chunk_lower = chunk.lower()
        
        # Compare with already selected chunks
        for existing in unique_chunks:
            existing_lower = existing.lower()
            
            # Calculate character overlap
            shorter_len = min(len(chunk_lower), len(existing_lower))
            longer_len = max(len(chunk_lower), len(existing_lower))
            
            if shorter_len == 0:
                continue
            
            # Count common characters (simple approach)
            common_chars = sum(1 for c1, c2 in zip(chunk_lower, existing_lower) if c1 == c2)
            overlap_ratio = common_chars / longer_len
            
            if overlap_ratio >= similarity_threshold:
                is_duplicate = True
                logger.debug(f"Chunk {i} is {overlap_ratio:.2f} similar to existing chunk - skipping")
                break
        
        if not is_duplicate:
            unique_chunks.append(chunk)
            unique_scores.append(scores[i])
    
    logger.info(f"Deduplication: {len(chunks)} → {len(unique_chunks)} chunks")
    
    return unique_chunks, unique_scores


def construct_grounded_context(chunks: List[str]) -> str:
    """
    Construct well-structured context from chunks with clear separators.
    
    Adds chunk numbering and separators to help the model
    understand distinct pieces of information and ground answers.
    
    Args:
        chunks: List of text chunks (already deduplicated)
        
    Returns:
        Formatted context string with chunk separators
        
    Notes:
        - Each chunk is clearly labeled (CHUNK 1, CHUNK 2, etc.)
        - Separators help prevent information blending
        - Preserves original chunk content exactly
    """
    if not chunks:
        return ""
    
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        # Add chunk separator and number
        context_parts.append(f"--- CHUNK {i} ---")
        context_parts.append(chunk.strip())
        context_parts.append("")  # Blank line between chunks
    
    return "\n".join(context_parts).strip()


# Request/Response models
class AskRequest(BaseModel):
    """Request body for ask endpoint"""
    question: str = Field(..., min_length=1, description="User's question")


class AskResponse(BaseModel):
    """Response body for ask endpoint"""
    answer: str = Field(..., description="Generated answer")
    source: str = Field(..., description="Source of answer: 'document' or 'web'")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    used_web: bool = Field(..., description="Whether web search was used")
    retrieved_chunks: int = Field(..., description="Number of chunks retrieved")
    web_results: Optional[List[Dict[str, str]]] = Field(None, description="Web search results if used")


@router.post("/ask/{doc_id}", response_model=AskResponse)
@limiter.limit("50/hour")  # 50 questions per hour
async def ask_question(
    request: Request,
    doc_id: int,
    question_data: AskRequest,
    current_user: User = Depends(get_current_user_or_dev_user)  # DEV MODE ONLY - Change back to get_current_user for production
) -> AskResponse:
    """
    Answer a question about a document using RAG.
    
    Pipeline:
    1. Validate document is indexed
    2. Embed query
    3. Retrieve similar chunks
    4. Compute confidence
    5. If confidence >= threshold: use document context
    6. If confidence < threshold: fall back to web search
    7. Generate answer using T5
    8. Return transparent response
    
    Args:
        doc_id: Document ID
        request: Question request
        current_user: Authenticated user
        
    Returns:
        AskResponse with answer, source, confidence, etc.
        
    Raises:
        HTTPException: If validation fails or processing errors occur
    """
    user_id = current_user.id
    question = question_data.question.strip()
    
    logger.info(f"Question for doc {doc_id} by user {user_id}: {question[:100]}...")
    
    # ========================================
    # STEP 1: VALIDATE DOCUMENT IS INDEXED
    # ========================================
    
    if not is_document_indexed(user_id, doc_id):
        logger.error(f"Document {doc_id} not indexed for user {user_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Document {doc_id} is not indexed. Please index the document first."
        )
    
    logger.debug("Document indexed ✓")
    
    # ========================================
    # STEP 2: VALIDATE QUESTION
    # ========================================
    
    if not question:
        logger.error("Empty question received")
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    logger.debug(f"Question: {question}")
    
    # ========================================
    # STEP 3: EMBED QUERY
    # ========================================
    
    try:
        query_embedding = embed_query(question)
    except Exception as e:
        logger.error(f"Query embedding failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to embed query: {str(e)}"
        )
    
    # Validate embedding shape (support both 384d and 768d models)
    if query_embedding.shape not in [(384,), (768,)]:
        logger.error(f"Invalid query embedding shape: {query_embedding.shape}")
        raise HTTPException(
            status_code=500,
            detail=f"Invalid query embedding dimension. Expected 384 or 768, got {query_embedding.shape}"
        )
    
    logger.debug(f"Query embedded ✓ (dimension={query_embedding.shape[0]})")
    
    # ========================================
    # STEP 4: GET DOCUMENT CHUNKS
    # ========================================
    
    try:
        chunks, _ = generate_document_embeddings(user_id, doc_id)
    except Exception as e:
        logger.error(f"Failed to get document chunks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load document chunks: {str(e)}"
        )
    
    if not chunks:
        logger.error(f"No chunks found for document {doc_id}")
        raise HTTPException(
            status_code=404,
            detail=f"No content found for document {doc_id}"
        )
    
    logger.debug(f"Loaded {len(chunks)} chunks")
    
    # ========================================
    # STEP 5: RETRIEVE SIMILAR CHUNKS
    # ========================================
    
    try:
        retrieved_chunks, scores = retrieve_similar_chunks(
            user_id=user_id,
            doc_id=doc_id,
            query_embedding=query_embedding,
            chunks=chunks,
            top_k=TOP_K_CHUNKS
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve similar chunks: {str(e)}"
        )
    
    if not retrieved_chunks:
        logger.error("No chunks retrieved")
        raise HTTPException(
            status_code=500,
            detail="No relevant chunks retrieved from document"
        )
    
    num_retrieved = len(retrieved_chunks)
    logger.debug(f"Retrieved {num_retrieved} chunks with scores: {scores}")
    
    # ========================================
    # STEP 5.5: DEDUPLICATE CHUNKS
    # ========================================
    
    try:
        retrieved_chunks, scores = deduplicate_chunks(retrieved_chunks, scores)
        num_unique = len(retrieved_chunks)
        logger.info(f"After deduplication: {num_unique} unique chunks (from {num_retrieved})")
    except Exception as e:
        logger.warning(f"Deduplication failed, using all chunks: {e}")
        # Continue with original chunks if deduplication fails
    
    # ========================================
    # STEP 6: EXTRACTIVE QA FROM DOCUMENT
    # ========================================
    
    logger.info("Starting extractive QA pipeline")
    
    try:
        extracted_answer, max_similarity, num_sentences = extract_answer_from_document(
            question=question,
            chunks=retrieved_chunks
        )
    except Exception as e:
        logger.error(f"Extractive QA failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract answer from document: {str(e)}"
        )
    
    logger.info(
        f"Extraction result: answer={'FOUND' if extracted_answer else 'NOT_FOUND'}, "
        f"max_similarity={max_similarity:.4f}, sentences={num_sentences}"
    )
    
    # ========================================
    # STEP 7: DECIDE SOURCE (DOCUMENT vs WEB)
    # ========================================
    
    web_results = None
    
    if extracted_answer:
        # DOCUMENT HAS ANSWER: Use extractive answer
        logger.info("Document contains answer → using extractive QA")
        
        # Compute confidence for extracted answer
        confidence = compute_extractive_confidence(max_similarity, num_sentences)
        
        # Use extracted answer directly (no LLM rewriting)
        # This preserves exact document wording and prevents degradation
        answer = extracted_answer
        
        logger.info(
            f"Using extracted answer (confidence: {confidence:.2f}, "
            f"similarity: {max_similarity:.2f}, sentences: {num_sentences})"
        )
        
        source = "document"
        used_web = False
        
    else:
        # DOCUMENT HAS NO ANSWER: Fall back to web (only if very low confidence)
        if max_similarity < 0.25:
            logger.warning(
                f"No document answer (max_similarity={max_similarity:.2f}) → attempting web fallback"
            )
            
            try:
                web_context, web_results = get_web_context_for_question(question)
                
                answer = generate_answer_from_context(question, web_context)
                
                # Low confidence for web fallback
                confidence = min(0.4, max_similarity + 0.15)  # Cap at 0.4 for web answers
                
                source = "web"
                used_web = True
                
                logger.info(f"Web search returned {len(web_results) if web_results else 0} results")
                
            except Exception as e:
                logger.error(f"Web fallback failed: {str(e)}")
                
                answer = (
                    "I could not find an answer in the document, "
                    "and web search is temporarily unavailable. "
                    "Please try again later."
                )
                
                # Very low confidence for failure case
                confidence = 0.0
                source = "none"
                used_web = False
                web_results = None
        else:
            # Edge case: similarity between 0.22-0.25, return low-confidence answer
            logger.info(
                f"Weak match (similarity={max_similarity:.2f}) → "
                "returning low-confidence document answer"
            )
            
            answer = "No clear answer found in the document."
            confidence = max_similarity
            source = "document"
            used_web = False
            web_results = None
    
    logger.info(f"Answer generated: {answer[:100]}...")
    
    # ========================================
    # STEP 8: RETURN RESPONSE
    # ========================================
    
    response = AskResponse(
        answer=answer,
        source=source,
        confidence=confidence,
        used_web=used_web,
        retrieved_chunks=num_retrieved,
        web_results=web_results
    )
    
    logger.info(
        f"Question answered: source={source}, confidence={confidence:.4f}, "
        f"used_web={used_web}, chunks={num_retrieved}"
    )
    
    return response
