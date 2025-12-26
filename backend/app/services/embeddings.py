# backend/app/services/embeddings.py

"""
STEP 6A: Embedding generation service using SentenceTransformers.

Production-grade embedding with:
- CPU-only mode (crash-safe on macOS ARM)
- Singleton pattern (load once)
- Thread-safe loading
- Memory-stable batching
- Comprehensive validation
"""

import os
# CRITICAL: Set BEFORE importing torch/transformers to prevent MPS crashes on macOS ARM
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import logging
from threading import Lock
from typing import List, Tuple
from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np

from app.core.config import get_settings
from app.utils.chunker import chunk_text, validate_chunks, get_chunk_stats

logger = logging.getLogger(__name__)

# Global singleton
_embedding_model: SentenceTransformer = None
_model_lock = Lock()

settings = get_settings()


def load_embedding_model() -> SentenceTransformer:
    """
    Load SentenceTransformer model (singleton, thread-safe).
    
    Configuration:
    - Model: sentence-transformers/all-mpnet-base-v2
    - Dimension: 768
    - Device: CPU (crash-safe)
    - Loads once per process
    
    Returns:
        SentenceTransformer model instance
        
    Raises:
        RuntimeError: If model loading fails
    """
    global _embedding_model
    
    with _model_lock:
        if _embedding_model is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            
            try:
                _embedding_model = SentenceTransformer(
                    settings.EMBEDDING_MODEL,
                    device='cpu'  # CRITICAL: CPU-only to prevent crashes
                )
                
                dim = _embedding_model.get_sentence_embedding_dimension()
                logger.info(f"Embedding model loaded: dimension={dim}")
                
                # Validate dimension (support both MiniLM-384d and MPNet-768d)
                if dim not in [384, 768]:
                    raise RuntimeError(f"Expected dimension 384 or 768, got {dim}")
                
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise RuntimeError(f"Embedding model loading failed: {str(e)}")
        
        return _embedding_model


def unload_model():
    """
    Unload embedding model from memory (Railway memory optimization).
    
    This is called before loading the summarizer to free ~80MB of RAM.
    The model will be automatically reloaded when needed.
    
    Used only in production (Railway) to prevent OOM crashes.
    In development, models stay loaded for better performance.
    """
    global _embedding_model
    
    with _model_lock:
        if _embedding_model is not None:
            logger.info("Unloading embedding model to free memory...")
            _embedding_model = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Embedding model unloaded successfully")


def generate_embeddings(chunks: List[str], batch_size: int = 8) -> np.ndarray:
    """
    Generate embeddings for text chunks.
    
    Rules:
    - Batch size: 8 (memory-stable)
    - L2 normalized (for cosine similarity)
    - Returns numpy array
    
    Args:
        chunks: List of text chunks
        batch_size: Batch size for encoding (default: 8, DO NOT INCREASE)
        
    Returns:
        Numpy array of shape (N, 768) where N = len(chunks)
        
    Raises:
        ValueError: If chunks is empty
        RuntimeError: If embedding generation fails
    """
    if not chunks:
        raise ValueError("Cannot generate embeddings for empty chunk list")
    
    logger.debug(f"Generating embeddings: {len(chunks)} chunks, batch_size={batch_size}")
    
    # Load model
    model = load_embedding_model()
    
    try:
        # Generate embeddings
        # CRITICAL: batch_size=8 to prevent memory issues
        # normalize_embeddings=True for L2 normalization (cosine similarity)
        # show_progress_bar=False for clean logs
        embeddings = model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Validate output
        if not isinstance(embeddings, np.ndarray):
            raise RuntimeError(f"Expected numpy array, got {type(embeddings)}")
        
        if embeddings.shape[0] != len(chunks):
            raise RuntimeError(
                f"Embedding count mismatch: expected {len(chunks)}, got {embeddings.shape[0]}"
            )
        
        # Support both 384d (MiniLM) and 768d (MPNet) dimensions
        dim = embeddings.shape[1]
        expected_dims = [384, 768]
        if dim not in expected_dims:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {expected_dims}, got {dim}"
            )
        
        logger.debug(f"Embeddings generated: shape={embeddings.shape}")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise RuntimeError(f"Failed to generate embeddings: {str(e)}")


def generate_document_embeddings(
    user_id: int,
    doc_id: int,
    max_words: int = 300,
    overlap_words: int = 80
) -> Tuple[List[str], np.ndarray]:
    """
    Generate embeddings for a document (Step 6A main function).
    
    Pipeline:
    1. Load cleaned.txt
    2. Chunk text (sentence-aware)
    3. Validate chunks
    4. Generate embeddings
    5. Return aligned (chunks, embeddings)
    
    Args:
        user_id: User ID
        doc_id: Document ID
        max_words: Maximum words per chunk (default: 300)
        overlap_words: Words to overlap (default: 80)
        
    Returns:
        Tuple of (chunks, embeddings) where:
        - chunks: List[str] of N text chunks
        - embeddings: np.ndarray of shape (N, 768)
        
    Raises:
        FileNotFoundError: If cleaned.txt does not exist
        ValueError: If no valid chunks generated
        RuntimeError: If embedding generation fails
    """
    logger.info(f"Starting embedding generation: user_id={user_id}, doc_id={doc_id}")
    
    # Step 1: Load cleaned.txt
    cleaned_path = settings.UPLOAD_DIR / str(user_id) / str(doc_id) / "cleaned.txt"
    
    if not cleaned_path.exists():
        logger.error(f"cleaned.txt not found: {cleaned_path}")
        raise FileNotFoundError(
            f"cleaned.txt not found for user {user_id}, doc {doc_id}. "
            f"Expected: {cleaned_path}"
        )
    
    try:
        cleaned_text = cleaned_path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Failed to read cleaned.txt: {e}")
        raise RuntimeError(f"Failed to read cleaned.txt: {str(e)}")
    
    if not cleaned_text or not cleaned_text.strip():
        raise ValueError(f"cleaned.txt is empty for user {user_id}, doc {doc_id}")
    
    logger.debug(f"Loaded cleaned.txt: {len(cleaned_text)} chars")
    
    # Step 2: Chunk text
    try:
        chunks = chunk_text(
            cleaned_text,
            max_words=max_words,
            overlap_words=overlap_words
        )
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        raise RuntimeError(f"Text chunking failed: {str(e)}")
    
    logger.debug(f"Chunking complete: {len(chunks)} raw chunks")
    
    # Step 3: Validate chunks
    chunks = validate_chunks(chunks, min_words=10)
    
    if not chunks:
        raise ValueError(
            f"No valid chunks generated for user {user_id}, doc {doc_id}. "
            f"Document may be too short or improperly formatted."
        )
    
    # Log chunk statistics
    stats = get_chunk_stats(chunks)
    logger.info(
        f"Chunks validated: {stats['num_chunks']} chunks, "
        f"avg {stats['avg_words_per_chunk']:.1f} words/chunk"
    )
    
    # Step 4: Generate embeddings
    try:
        embeddings = generate_embeddings(chunks, batch_size=8)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
    
    # Step 5: Verify alignment
    if len(chunks) != embeddings.shape[0]:
        raise RuntimeError(
            f"Chunk-embedding mismatch: {len(chunks)} chunks, {embeddings.shape[0]} embeddings"
        )
    
    logger.info(
        f"Embedding generation complete: user_id={user_id}, doc_id={doc_id}, "
        f"chunks={len(chunks)}, embeddings shape={embeddings.shape}"
    )
    
    return chunks, embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Generate embedding for a single query string.
    
    Used for semantic search and Q&A.
    
    Args:
        query: Query text
        
    Returns:
        Numpy array of shape (768,)
        
    Raises:
        ValueError: If query is empty
        RuntimeError: If embedding fails
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    query = query.strip()
    
    try:
        embeddings = generate_embeddings([query], batch_size=1)
        return embeddings[0]  # Return 1D array
    except Exception as e:
        logger.error(f"Query embedding failed: {e}")
        raise RuntimeError(f"Failed to embed query: {str(e)}")


def get_embedding_dimension() -> int:
    """
    Get embedding model dimension.
    
    Returns:
        Embedding dimension (768 for all-mpnet-base-v2)
    """
    model = load_embedding_model()
    return model.get_sentence_embedding_dimension()
