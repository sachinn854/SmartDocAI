# backend/app/utils/chunker.py

"""
STEP 6A: Sentence-aware text chunking for embeddings.

Production-grade chunking with:
- Sentence boundary preservation
- Deterministic output
- Memory efficiency
- Comprehensive validation
"""

import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.
    
    Handles:
    - Period, question mark, exclamation point
    - Preserves sentence boundaries
    - Filters empty sentences
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []
    
    # Split on sentence boundaries: . ! ? followed by space and capital
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text)
    
    # Clean and filter
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Input text
        
    Returns:
        Word count
    """
    return len(text.split())


def chunk_text(
    text: str,
    max_words: int = 300,
    overlap_words: int = 80
) -> List[str]:
    """
    Chunk text into overlapping segments while preserving sentence boundaries.
    
    Rules:
    - Maximum 300 words per chunk
    - 80-word overlap between chunks
    - Never split mid-sentence
    - Deterministic (same input → same output)
    
    Args:
        text: Input text to chunk
        max_words: Maximum words per chunk (default: 300)
        overlap_words: Words to overlap between chunks (default: 80)
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If text is empty
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text")
    
    text = text.strip()
    total_words = count_words(text)
    
    logger.debug(f"Chunking text: {total_words} words, max={max_words}, overlap={overlap_words}")
    
    # If text fits in one chunk, return as-is
    if total_words <= max_words:
        return [text]
    
    # Split into sentences
    sentences = split_into_sentences(text)
    
    if not sentences:
        # Fallback: no sentence boundaries found, split by words
        logger.warning("No sentence boundaries found, falling back to word-based chunking")
        return _chunk_by_words(text, max_words, overlap_words)
    
    chunks = []
    current_chunk_sentences = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = count_words(sentence)
        
        # Handle sentence longer than max_words
        if sentence_words > max_words:
            # Save current chunk if exists
            if current_chunk_sentences:
                chunks.append(' '.join(current_chunk_sentences))
                current_chunk_sentences = []
                current_word_count = 0
            
            # Split long sentence by words
            long_sentence_chunks = _chunk_by_words(sentence, max_words, overlap_words)
            chunks.extend(long_sentence_chunks)
            continue
        
        # Check if adding sentence exceeds max_words
        if current_word_count + sentence_words > max_words:
            # Save current chunk
            chunks.append(' '.join(current_chunk_sentences))
            
            # Create overlap for next chunk
            overlap_sentences = _get_overlap_sentences(
                current_chunk_sentences,
                overlap_words
            )
            
            # Start new chunk with overlap
            current_chunk_sentences = overlap_sentences + [sentence]
            current_word_count = sum(count_words(s) for s in current_chunk_sentences)
        else:
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_word_count += sentence_words
    
    # Add final chunk if exists
    if current_chunk_sentences:
        chunks.append(' '.join(current_chunk_sentences))
    
    logger.debug(f"Chunking complete: {len(chunks)} chunks generated")
    
    return chunks


def _chunk_by_words(text: str, max_words: int, overlap_words: int) -> List[str]:
    """
    Fallback: chunk text by words when sentence boundaries unavailable.
    
    Args:
        text: Input text
        max_words: Maximum words per chunk
        overlap_words: Words to overlap
        
    Returns:
        List of word-based chunks
    """
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk_words = words[i:i + max_words]
        chunks.append(' '.join(chunk_words))
        
        # Move forward by (max_words - overlap_words)
        i += max(1, max_words - overlap_words)
    
    return chunks


def _get_overlap_sentences(sentences: List[str], overlap_words: int) -> List[str]:
    """
    Get last N sentences that fit within overlap_words limit.
    
    Args:
        sentences: List of sentences
        overlap_words: Maximum words for overlap
        
    Returns:
        List of sentences for overlap
    """
    if not sentences or overlap_words <= 0:
        return []
    
    overlap_sentences = []
    word_count = 0
    
    # Work backwards from end
    for sentence in reversed(sentences):
        sentence_words = count_words(sentence)
        
        if word_count + sentence_words <= overlap_words:
            overlap_sentences.insert(0, sentence)
            word_count += sentence_words
        else:
            break
    
    return overlap_sentences


def validate_chunks(chunks: List[str], min_words: int = 10) -> List[str]:
    """
    Validate and filter chunks.
    
    Removes:
    - Empty chunks
    - Chunks with < min_words
    - Duplicate chunks
    
    Args:
        chunks: List of chunks to validate
        min_words: Minimum words per chunk (default: 10)
        
    Returns:
        Filtered list of valid chunks
    """
    if not chunks:
        return []
    
    valid_chunks = []
    seen_chunks = set()
    
    for chunk in chunks:
        chunk = chunk.strip()
        
        # Skip empty
        if not chunk:
            continue
        
        # Skip too short
        if count_words(chunk) < min_words:
            logger.debug(f"Skipping chunk with < {min_words} words")
            continue
        
        # Skip duplicates
        if chunk in seen_chunks:
            logger.debug("Skipping duplicate chunk")
            continue
        
        valid_chunks.append(chunk)
        seen_chunks.add(chunk)
    
    logger.debug(f"Validation: {len(chunks)} → {len(valid_chunks)} valid chunks")
    
    return valid_chunks


def get_chunk_stats(chunks: List[str]) -> Dict[str, any]:
    """
    Get statistics about chunks.
    
    Args:
        chunks: List of chunks
        
    Returns:
        Dictionary with statistics
    """
    if not chunks:
        return {
            'num_chunks': 0,
            'total_words': 0,
            'avg_words_per_chunk': 0,
            'min_words': 0,
            'max_words': 0
        }
    
    word_counts = [count_words(chunk) for chunk in chunks]
    
    return {
        'num_chunks': len(chunks),
        'total_words': sum(word_counts),
        'avg_words_per_chunk': sum(word_counts) / len(word_counts),
        'min_words': min(word_counts),
        'max_words': max(word_counts)
    }
