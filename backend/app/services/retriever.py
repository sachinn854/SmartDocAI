# backend/app/services/retriever.py

"""
STEP 6B: FAISS-based vector indexing and semantic retrieval.

Production-grade FAISS pipeline with:
- Exact similarity search (IndexFlatIP)
- Persistent disk storage
- Safe index lifecycle management
- Comprehensive validation
- Crash-free operation
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict

import faiss
import numpy as np

from app.core.config import get_settings
from app.services.embeddings import generate_document_embeddings

logger = logging.getLogger(__name__)

settings = get_settings()


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS index from embeddings.
    
    Uses IndexFlatIP (Inner Product) for exact cosine similarity search
    on L2-normalized vectors.
    
    Args:
        embeddings: Numpy array of shape (N, 768) with L2-normalized vectors
        
    Returns:
        FAISS IndexFlatIP ready for search
        
    Raises:
        ValueError: If embeddings is invalid
        RuntimeError: If index creation fails
    """
    # Validate input
    if not isinstance(embeddings, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(embeddings)}")
    
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings.shape}")
    
    # Support both 384d (MiniLM) and 768d (MPNet) dimensions
    dim = embeddings.shape[1]
    if dim not in [384, 768]:
        raise ValueError(f"Expected dimension 384 or 768, got {dim}")
    
    if embeddings.shape[0] == 0:
        raise ValueError("Cannot build index from empty embeddings")
    
    num_vectors, dimension = embeddings.shape
    
    logger.debug(f"Building FAISS index: {num_vectors} vectors, dimension={dimension}")
    
    try:
        # Create IndexFlatIP for exact cosine similarity
        # For L2-normalized vectors: cosine(a,b) = dot(a,b) = inner_product(a,b)
        index = faiss.IndexFlatIP(dimension)
        
        # Convert to float32 (FAISS requirement)
        embeddings_f32 = embeddings.astype(np.float32)
        
        # Add vectors to index
        index.add(embeddings_f32)
        
        # Verify index size
        if index.ntotal != num_vectors:
            raise RuntimeError(
                f"Index size mismatch: expected {num_vectors}, got {index.ntotal}"
            )
        
        logger.info(f"FAISS index built: {index.ntotal} vectors")
        
        return index
        
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        raise RuntimeError(f"FAISS index creation failed: {str(e)}")


def save_faiss_index(index: faiss.Index, user_id: int, doc_id: int) -> Path:
    """
    Save FAISS index to disk with proper directory structure.
    
    Storage path: backend/app/data/index/{user_id}/{doc_id}/faiss.index
    
    Args:
        index: FAISS index to save
        user_id: User ID for isolation
        doc_id: Document ID
        
    Returns:
        Path to saved index file
        
    Raises:
        RuntimeError: If save fails
    """
    # Construct index directory
    index_dir = settings.INDEX_DIR / str(user_id) / str(doc_id)
    index_path = index_dir / "faiss.index"
    
    # Check if already exists (fail-fast)
    if index_path.exists():
        logger.warning(f"Index already exists, overwriting: {index_path}")
    
    try:
        # Create directory structure
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save index to disk
        faiss.write_index(index, str(index_path))
        
        logger.info(f"FAISS index saved: {index_path}")
        
        return index_path
        
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}")
        raise RuntimeError(f"Failed to save index to {index_path}: {str(e)}")


def load_faiss_index(user_id: int, doc_id: int) -> faiss.Index:
    """
    Load FAISS index from disk.
    
    Args:
        user_id: User ID
        doc_id: Document ID
        
    Returns:
        Loaded FAISS index
        
    Raises:
        FileNotFoundError: If index file doesn't exist
        RuntimeError: If load fails or index is empty
    """
    # Construct index path
    index_path = settings.INDEX_DIR / str(user_id) / str(doc_id) / "faiss.index"
    
    # Check if exists
    if not index_path.exists():
        logger.error(f"Index file not found: {index_path}")
        raise FileNotFoundError(
            f"FAISS index not found for user {user_id}, doc {doc_id}. "
            f"Expected: {index_path}. "
            f"Run index_document() first."
        )
    
    try:
        # Load index from disk
        index = faiss.read_index(str(index_path))
        
        # Validate loaded index
        if index.ntotal == 0:
            raise RuntimeError(f"Loaded index is empty: {index_path}")
        
        logger.info(f"FAISS index loaded: {index_path}, vectors={index.ntotal}")
        
        return index
        
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        raise RuntimeError(f"Failed to load index from {index_path}: {str(e)}")


def is_document_indexed(user_id: int, doc_id: int) -> bool:
    """
    Check if document has a FAISS index.
    
    Args:
        user_id: User ID
        doc_id: Document ID
        
    Returns:
        True if index exists, False otherwise
    """
    index_path = settings.INDEX_DIR / str(user_id) / str(doc_id) / "faiss.index"
    return index_path.exists() and index_path.is_file()


def index_document(user_id: int, doc_id: int) -> Dict:
    """
    Index a document by generating embeddings and building FAISS index.
    
    Complete pipeline:
    1. Call Step 6A: generate_document_embeddings()
    2. Validate embeddings
    3. Build FAISS index
    4. Save index to disk
    5. Return metadata
    
    Args:
        user_id: User ID
        doc_id: Document ID
        
    Returns:
        Dictionary with indexing metadata:
        {
            "doc_id": int,
            "vectors": int,
            "dimension": int,
            "status": "indexed"
        }
        
    Raises:
        FileNotFoundError: If cleaned.txt doesn't exist
        ValueError: If no valid chunks
        RuntimeError: If indexing fails
    """
    logger.info(f"Starting document indexing: user_id={user_id}, doc_id={doc_id}")
    
    # Step 1: Generate embeddings (calls Step 6A)
    try:
        chunks, embeddings = generate_document_embeddings(
            user_id=user_id,
            doc_id=doc_id
        )
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
    
    # Step 2: Validate embeddings
    if not isinstance(embeddings, np.ndarray):
        raise RuntimeError(f"Expected numpy array, got {type(embeddings)}")
    
    if embeddings.ndim != 2:
        raise RuntimeError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    
    # Support both 384d (MiniLM) and 768d (MPNet) dimensions
    dim = embeddings.shape[1]
    if dim not in [384, 768]:
        raise RuntimeError(f"Expected dimension 384 or 768, got {dim}")
    
    if len(chunks) != embeddings.shape[0]:
        raise RuntimeError(
            f"Chunk-embedding mismatch: {len(chunks)} chunks, {embeddings.shape[0]} embeddings"
        )
    
    num_vectors = embeddings.shape[0]
    
    logger.info(f"Embeddings validated: {num_vectors} vectors, dimension={dim}")
    
    # Step 3: Build FAISS index
    try:
        index = build_faiss_index(embeddings)
    except Exception as e:
        logger.error(f"FAISS index build failed: {e}")
        raise RuntimeError(f"Failed to build FAISS index: {str(e)}")
    
    # Step 4: Save index to disk
    try:
        index_path = save_faiss_index(index, user_id, doc_id)
    except Exception as e:
        logger.error(f"Index save failed: {e}")
        raise RuntimeError(f"Failed to save index: {str(e)}")
    
    # Step 5: Return metadata
    metadata = {
        "doc_id": doc_id,
        "vectors": num_vectors,
        "dimension": dim,
        "status": "indexed",
        "index_path": str(index_path)
    }
    
    logger.info(f"Document indexed successfully: {metadata}")
    
    return metadata


def retrieve_similar_chunks(
    user_id: int,
    doc_id: int,
    query_embedding: np.ndarray,
    chunks: List[str],
    top_k: int = 5
) -> Tuple[List[str], List[float]]:
    """
    Retrieve most similar chunks using FAISS semantic search.
    
    Pipeline:
    1. Load FAISS index
    2. Validate query embedding
    3. Perform similarity search
    4. Map indices to chunks
    5. Return chunks + scores
    
    Args:
        user_id: User ID
        doc_id: Document ID
        query_embedding: Query embedding of shape (768,)
        chunks: List of text chunks (aligned with index)
        top_k: Number of chunks to retrieve (default: 5)
        
    Returns:
        Tuple of:
        - List[str]: Retrieved chunks (most similar first)
        - List[float]: Similarity scores (descending order)
        
    Raises:
        FileNotFoundError: If index doesn't exist
        ValueError: If query embedding is invalid
        RuntimeError: If retrieval fails
    """
    logger.debug(f"Retrieving similar chunks: user_id={user_id}, doc_id={doc_id}, top_k={top_k}")
    
    # Step 1: Load FAISS index
    try:
        index = load_faiss_index(user_id, doc_id)
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
        raise
    
    # Step 2: Validate query embedding
    if not isinstance(query_embedding, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(query_embedding)}")
    
    # Support both 384d (MiniLM) and 768d (MPNet) dimensions
    dim = query_embedding.shape[0] if query_embedding.ndim == 1 else query_embedding.shape[1]
    if dim not in [384, 768]:
        raise ValueError(f"Expected dimension 384 or 768, got {dim}")
    
    # Validate chunks count
    if len(chunks) != index.ntotal:
        raise ValueError(
            f"Chunks count mismatch: {len(chunks)} chunks, {index.ntotal} vectors in index"
        )
    
    # Adjust top_k if necessary
    effective_top_k = min(top_k, index.ntotal)
    
    # Step 3: Reshape query embedding for FAISS (needs 2D)
    query_2d = query_embedding.reshape(1, -1).astype(np.float32)
    
    try:
        # Perform similarity search
        # Returns: distances (inner products), indices
        scores, indices = index.search(query_2d, effective_top_k)
        
        # Extract results (search returns batch, we have batch_size=1)
        scores = scores[0]  # Shape: (top_k,)
        indices = indices[0]  # Shape: (top_k,)
        
    except Exception as e:
        logger.error(f"FAISS search failed: {e}")
        raise RuntimeError(f"Semantic search failed: {str(e)}")
    
    # Step 4: Map indices to chunks
    retrieved_chunks = []
    retrieved_scores = []
    
    for idx, score in zip(indices, scores):
        # Validate index
        if idx < 0 or idx >= len(chunks):
            logger.warning(f"Invalid index {idx}, skipping")
            continue
        
        retrieved_chunks.append(chunks[idx])
        retrieved_scores.append(float(score))
    
    logger.debug(f"Retrieved {len(retrieved_chunks)} chunks")
    
    return retrieved_chunks, retrieved_scores


def get_index_stats(user_id: int, doc_id: int) -> Dict:
    """
    Get statistics about a FAISS index.
    
    Args:
        user_id: User ID
        doc_id: Document ID
        
    Returns:
        Dictionary with index statistics
        
    Raises:
        FileNotFoundError: If index doesn't exist
    """
    index = load_faiss_index(user_id, doc_id)
    index_path = settings.INDEX_DIR / str(user_id) / str(doc_id) / "faiss.index"
    
    # Get file size
    file_size_bytes = index_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    return {
        "num_vectors": index.ntotal,
        "dimension": index.d,
        "index_type": type(index).__name__,
        "file_size_mb": round(file_size_mb, 2),
        "index_path": str(index_path)
    }
