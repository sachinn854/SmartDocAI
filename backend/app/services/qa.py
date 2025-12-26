# backend/app/services/qa.py

"""
STEP 7: Extractive Question Answering with Optional LLM Simplification.

Document-first extractive QA pipeline:
- Sentence-level extraction from document chunks
- Similarity-based sentence selection
- Optional T5 simplification (rewrite only, no generation)
- Web fallback only when document has no answer
- CPU-only mode (no MPS crashes)
- Singleton pattern for model loading
"""

import os
import logging
import threading
import re
from typing import Optional, List, Tuple

# CRITICAL: Set environment variables BEFORE importing torch/transformers
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

logger = logging.getLogger(__name__)

# Global model cache (singleton)
_qa_model: Optional[T5ForConditionalGeneration] = None
_qa_tokenizer: Optional[T5Tokenizer] = None
_embedding_model: Optional[SentenceTransformer] = None
_qa_model_lock = threading.Lock()
_embedding_lock = threading.Lock()

# Import settings to use auto-optimized models
from app.core.config import get_settings

settings = get_settings()

# T5 Configuration (for optional simplification)
QA_MODEL_NAME = settings.SUMMARIZER_MODEL  # Uses auto-optimized model (flan-t5-small in prod)
MAX_INPUT_LENGTH = 512  # T5 max input
MAX_OUTPUT_LENGTH = 120  # Reasonable answer length
DEVICE = "cpu"  # CPU-only to prevent MPS crashes

# Extractive QA Configuration
# CRITICAL: Use SAME model as document indexing (embeddings.py) for consistency
EMBEDDING_MODEL_NAME = settings.EMBEDDING_MODEL  # Uses auto-optimized model (MiniLM in prod)
MIN_SENTENCE_SIMILARITY = 0.18  # Lowered threshold for better recall
MAX_SENTENCES_PER_ANSWER = 5    # Maximum 5 sentences (flexible length)
MIN_SENTENCES_PER_ANSWER = 1    # Allow single-sentence answers
MIN_ANSWER_LENGTH = 100         # Minimum characters (~2-3 lines)
MAX_ANSWER_LENGTH = 600         # Maximum characters (~12 lines at 50 chars/line)


def load_qa_model() -> tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """
    Load T5-small model and tokenizer (singleton).
    
    Uses thread-safe lazy loading pattern.
    Model is loaded once and cached globally.
    
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        RuntimeError: If model loading fails
    """
    global _qa_model, _qa_tokenizer
    
    # Fast path: model already loaded
    if _qa_model is not None and _qa_tokenizer is not None:
        logger.debug("Using cached T5 model")
        return _qa_model, _qa_tokenizer
    
    # Thread-safe model loading
    with _qa_model_lock:
        # Double-check after acquiring lock
        if _qa_model is not None and _qa_tokenizer is not None:
            return _qa_model, _qa_tokenizer
        
        logger.info(f"Loading T5 model: {QA_MODEL_NAME}")
        
        try:
            # Load tokenizer
            tokenizer = T5Tokenizer.from_pretrained(QA_MODEL_NAME)
            
            # Load model
            model = T5ForConditionalGeneration.from_pretrained(QA_MODEL_NAME)
            
            # Move to CPU (explicit)
            model = model.to(DEVICE)
            
            # Set to evaluation mode
            model.eval()
            
            # Cache globally
            _qa_model = model
            _qa_tokenizer = tokenizer
            
            logger.info(f"T5 model loaded successfully: {QA_MODEL_NAME}, device={DEVICE}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load T5 model: {e}")
            raise RuntimeError(f"T5 model loading failed: {str(e)}")


def load_embedding_model() -> SentenceTransformer:
    """
    Load SentenceTransformer model for sentence embeddings (singleton).
    
    Uses thread-safe lazy loading pattern.
    Model is loaded once and cached globally.
    
    Returns:
        SentenceTransformer model
        
    Raises:
        RuntimeError: If model loading fails
    """
    global _embedding_model
    
    # Fast path: model already loaded
    if _embedding_model is not None:
        return _embedding_model
    
    # Slow path: load model (thread-safe)
    with _embedding_lock:
        # Double-check after acquiring lock
        if _embedding_model is not None:
            return _embedding_model
        
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        
        try:
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            model.to(DEVICE)
            
            _embedding_model = model
            
            logger.info(f"Embedding model loaded successfully (device={DEVICE})")
            
            return _embedding_model
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model: {str(e)}")


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences with improved abbreviation handling.
    
    Handles common sentence boundaries:
    - Period followed by space and capital letter
    - Question marks and exclamation points
    - Preserves decimal numbers (e.g., 3.14)
    - Protects common abbreviations (Dr., U.S., e.g., etc.)
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentence strings
    """
    if not text or not text.strip():
        return []
    
    # Abbreviation mapping for protection during splitting
    abbrev_map = {
        'Dr.': 'Dr__TEMP__',
        'Mr.': 'Mr__TEMP__',
        'Mrs.': 'Mrs__TEMP__',
        'Ms.': 'Ms__TEMP__',
        'U.S.': 'US__TEMP__',
        'U.K.': 'UK__TEMP__',
        'e.g.': 'eg__TEMP__',
        'i.e.': 'ie__TEMP__',
        'etc.': 'etc__TEMP__',
        'vs.': 'vs__TEMP__',
        'Ph.D.': 'PhD__TEMP__'
    }
    
    # Replace abbreviations with temporary markers
    protected_text = text
    for abbrev, temp in abbrev_map.items():
        protected_text = protected_text.replace(abbrev, temp)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected_text.strip())
    
    # Restore abbreviations
    restore_map = {v: k for k, v in abbrev_map.items()}
    restored_sentences = []
    for s in sentences:
        for temp, abbrev in restore_map.items():
            s = s.replace(temp, abbrev)
        restored_sentences.append(s)
    
    # Clean and filter (minimum 15 chars for better quality)
    sentences = [s.strip() for s in restored_sentences if s.strip() and len(s) >= 15]
    
    logger.debug(f"Split into {len(sentences)} sentences")
    
    return sentences


def extract_sentences_from_chunks(chunks: List[str]) -> List[str]:
    """
    Extract all sentences from retrieved chunks.
    
    Deduplicates sentences to avoid repetition.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        List of unique sentences
    """
    all_sentences = []
    seen = set()
    
    for chunk in chunks:
        sentences = split_into_sentences(chunk)
        for sentence in sentences:
            # Deduplicate by normalized text
            normalized = sentence.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                all_sentences.append(sentence)
    
    logger.debug(f"Extracted {len(all_sentences)} unique sentences from {len(chunks)} chunks")
    
    return all_sentences


def compute_sentence_similarities(question: str, sentences: List[str]) -> List[Tuple[str, float]]:
    """
    Compute similarity between question and each sentence.
    
    Uses sentence embeddings and cosine similarity.
    
    Args:
        question: User's question
        sentences: List of candidate sentences
        
    Returns:
        List of (sentence, similarity_score) tuples, sorted by score (descending)
        
    Raises:
        RuntimeError: If embedding fails
    """
    if not sentences:
        return []
    
    try:
        # Load embedding model
        model = load_embedding_model()
        
        # Embed question
        question_embedding = model.encode(question, convert_to_numpy=True)
        question_embedding = question_embedding / np.linalg.norm(question_embedding)
        
        # Embed all sentences
        sentence_embeddings = model.encode(sentences, convert_to_numpy=True)
        
        # Normalize sentence embeddings
        norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        sentence_embeddings = sentence_embeddings / norms
        
        # Compute cosine similarities
        similarities = np.dot(sentence_embeddings, question_embedding)
        
        # Create (sentence, score) pairs
        scored_sentences = list(zip(sentences, similarities.tolist()))
        
        # Sort by score (descending)
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Computed similarities for {len(sentences)} sentences")
        if scored_sentences:
            logger.debug(f"Top score: {scored_sentences[0][1]:.4f}, Bottom score: {scored_sentences[-1][1]:.4f}")
        
        return scored_sentences
        
    except Exception as e:
        logger.error(f"Sentence similarity computation failed: {e}")
        raise RuntimeError(f"Failed to compute sentence similarities: {str(e)}")


def extract_answer_from_document(question: str, chunks: List[str]) -> Tuple[Optional[str], float, int]:
    """
    Extract answer from document using sentence-level similarity.
    
    IMPROVED EXTRACTIVE APPROACH:
    1. Split chunks into sentences with abbreviation handling
    2. Compute similarity between question and each sentence
    3. Select top sentences above threshold with flexible length control
    4. Concatenate as extracted answer (variable length: 100-600 chars)
    
    Args:
        question: User's question
        chunks: Retrieved document chunks
        
    Returns:
        Tuple of (extracted_answer, max_similarity, num_sentences)
        - extracted_answer: Concatenated sentences or None if no match
        - max_similarity: Highest sentence similarity score
        - num_sentences: Number of sentences in extracted answer
    """
    logger.info(f"Extractive QA: question='{question[:50]}...', chunks={len(chunks)}")
    
    # Step 1: Extract sentences from chunks
    sentences = extract_sentences_from_chunks(chunks)
    
    if not sentences:
        logger.warning("No sentences found in chunks")
        return None, 0.0, 0
    
    logger.info(f"Extracted {len(sentences)} unique sentences from chunks")
    
    # Step 2: Compute similarities
    try:
        scored_sentences = compute_sentence_similarities(question, sentences)
    except Exception as e:
        logger.error(f"Similarity computation failed: {e}")
        return None, 0.0, 0
    
    if not scored_sentences:
        logger.warning("No scored sentences")
        return None, 0.0, 0
    
    # Step 3: Filter by threshold
    relevant_sentences = [
        (sent, score) for sent, score in scored_sentences 
        if score >= MIN_SENTENCE_SIMILARITY
    ]
    
    # DIAGNOSTIC LOGGING - helps identify threshold issues
    logger.info(
        f"Relevant: {len(relevant_sentences)}/{len(scored_sentences)} sentences "
        f"(threshold={MIN_SENTENCE_SIMILARITY})"
    )
    if scored_sentences:
        top_5_scores = [f"{s[1]:.3f}" for s in scored_sentences[:5]]
        logger.info(f"Top 5 similarity scores: {top_5_scores}")
    
    if not relevant_sentences:
        max_score = scored_sentences[0][1] if scored_sentences else 0.0
        logger.warning(f"No sentences above threshold (max similarity: {max_score:.4f})")
        return None, max_score, 0
    
    # Step 4: Smart sentence selection with flexible length control
    selected_sentences = []
    total_length = 0
    
    for sent, score in relevant_sentences:
        sentence_length = len(sent)
        
        # Stop if max sentences reached
        if len(selected_sentences) >= MAX_SENTENCES_PER_ANSWER:
            logger.debug(f"Reached max sentences limit ({MAX_SENTENCES_PER_ANSWER})")
            break
        
        # Calculate new length if we add this sentence
        new_length = total_length + sentence_length + (1 if total_length > 0 else 0)
        
        # Check if adding would exceed max length
        if new_length > MAX_ANSWER_LENGTH:
            # Only stop if we already have enough content
            if total_length >= MIN_ANSWER_LENGTH:
                logger.debug(f"Reached sufficient length ({total_length} chars)")
                break
            # Otherwise, take this sentence even if slightly over
            # (better to be complete than cut off mid-thought)
        
        selected_sentences.append((sent, score))
        total_length = new_length
        logger.debug(f"Selected sentence {len(selected_sentences)}: {sent[:60]}... (score: {score:.3f})")
    
    # Check if we have any sentences
    if not selected_sentences:
        logger.warning("No sentences selected after filtering")
        return None, 0.0, 0
    
    max_similarity = selected_sentences[0][1]
    
    # Check minimum sentence count (only enforce if we have very low count)
    if len(selected_sentences) < MIN_SENTENCES_PER_ANSWER:
        logger.info(f"Only {len(selected_sentences)} relevant sentences (min: {MIN_SENTENCES_PER_ANSWER})")
        return None, max_similarity, len(selected_sentences)
    
    # Step 5: Concatenate sentences
    extracted_answer = " ".join([sent for sent, _ in selected_sentences])
    
    # Final safety check: If way too long, truncate gracefully at sentence boundary
    if len(extracted_answer) > MAX_ANSWER_LENGTH + 50:  # +50 chars tolerance
        # Try to truncate at last complete sentence
        truncated = extracted_answer[:MAX_ANSWER_LENGTH]
        last_period = truncated.rfind('. ')
        if last_period > MAX_ANSWER_LENGTH // 2:  # Only if we keep at least half
            extracted_answer = truncated[:last_period + 1]
        else:
            # Truncate at word boundary
            extracted_answer = truncated.rsplit(' ', 1)[0]
            if not extracted_answer.endswith('.'):
                extracted_answer += '...'
        logger.debug(f"Truncated answer to {len(extracted_answer)} chars")
    
    logger.info(
        f"✓ Answer extracted: {len(selected_sentences)} sentences, "
        f"{len(extracted_answer)} chars, "
        f"max_similarity={max_similarity:.3f}"
    )
    
    return extracted_answer, max_similarity, len(selected_sentences)


def simplify_answer_with_llm(extracted_answer: str, question: str) -> Optional[str]:
    """
    Optionally simplify extracted answer using T5 LLM.
    
    REWRITE ONLY - no new information added.
    
    Args:
        extracted_answer: Raw extracted text from document
        question: Original user question (for context)
        
    Returns:
        Simplified answer or None if simplification fails
        
    Notes:
        - This is OPTIONAL - if it fails, return extracted answer as-is
        - LLM is instructed to REWRITE only, not add information
        - Fallback: return None on any error
    """
    logger.info("Attempting optional LLM simplification")
    
    try:
        # Load model (singleton)
        model, tokenizer = load_qa_model()
        
        # Simplification prompt (REWRITE ONLY)
        instruction = (
            "Rewrite the following answer in clear, simple language. "
            "Do NOT add new information. "
            "Do NOT remove facts. "
            "Keep the answer between 2-4 sentences."
        )
        
        prompt = f"{instruction}\n\nQuestion: {question}\n\nExtracted Answer: {extracted_answer}\n\nSimplified Answer:"
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate simplified version
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=120,
                min_new_tokens=20,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.7,
                do_sample=False
            )
        
        simplified = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        simplified = simplified.strip()
        
        if simplified and len(simplified) > 20:
            logger.info(f"LLM simplification successful ({len(simplified)} chars)")
            return simplified
        else:
            logger.warning("LLM returned empty/short simplification")
            return None
            
    except Exception as e:
        logger.warning(f"LLM simplification failed (will use extracted answer): {e}")
        return None


def generate_answer_from_context(question: str, context: str) -> str:
    """
    DEPRECATED: Legacy function for backward compatibility.
    
    This function is kept for web fallback logic.
    For document-based QA, use extract_answer_from_document instead.
    
    Uses T5 with grounding-focused prompt to ensure answers are based
    only on the provided context, not external knowledge.
    
    Args:
        question: User's question (required)
        context: Retrieved text context (required)
        
    Returns:
        Generated answer string
        
    Raises:
        ValueError: If question or context is empty
        RuntimeError: If generation fails
    """
    # Validate inputs
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    if not context or not context.strip():
        raise ValueError("Context cannot be empty")
    
    question = question.strip()
    context = context.strip()
    
    logger.debug(f"Generating answer (legacy mode) for question: {question[:100]}...")
    
    # Load model (singleton)
    try:
        model, tokenizer = load_qa_model()
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise RuntimeError(f"Failed to load T5 model: {str(e)}")
    
    # Format prompt with strict grounding instruction
    instruction = (
        "Answer the question using ONLY the information provided in the context below. "
        "If the answer is not contained in the context, state that the document does not provide this information. "
        "Provide a clear, complete answer in 2-4 sentences when information is available."
    )
    
    prompt = f"{instruction}\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    logger.debug(f"Prompt length: {len(prompt)} chars")
    
    try:
        # Tokenize input with truncation to fit T5 limits
        inputs = tokenizer(
            prompt,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move inputs to CPU
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Generate answer with parameters optimized for grounded, detailed responses
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=150,  # Increased from 120 for more detailed answers
                min_new_tokens=30,   # Reduced from 40 to allow concise answers when appropriate
                num_beams=4,         # Beam search for better quality
                early_stopping=True,
                no_repeat_ngram_size=3,  # Prevent repetition
                temperature=0.7,
                do_sample=False      # Deterministic output for consistency
            )
        
        # Decode answer
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Clean answer
        answer = answer.strip()
        
        if not answer:
            logger.warning("Generated empty answer")
            return "I couldn't generate an answer based on the provided context."
        
        logger.debug(f"Generated answer: {answer[:100]}...")
        
        return answer
        
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        raise RuntimeError(f"Failed to generate answer: {str(e)}")


def compute_confidence(scores: list[float]) -> float:
    """
    DEPRECATED: Legacy confidence for chunk-based retrieval.
    
    Kept for backward compatibility with web fallback.
    For extractive QA, use compute_extractive_confidence instead.
    
    Enhanced confidence calculation that:
    1. Filters out low-quality chunks (below min_threshold)
    2. Weights by number of high-quality chunks
    3. Normalizes to [0, 1] range
    
    Args:
        scores: List of FAISS similarity scores (descending order)
        
    Returns:
        Confidence score (0.0 to 1.0)
        
    Notes:
        - Returns 0.0 if no scores or all below threshold
        - Confidence = (average of high-quality scores) * (quality_ratio^0.5)
        - quality_ratio = (num_high_quality / total_chunks)
        - This balances score quality with chunk quantity
    """
    MIN_THRESHOLD = 0.5  # Minimum score to consider a chunk relevant
    
    # ISSUE 2 FIX: Return 0.0 for empty scores instead of raising exception
    if not scores:
        return 0.0
    
    # Filter high-quality chunks (above threshold)
    high_quality_scores = [s for s in scores if s >= MIN_THRESHOLD]
    
    if not high_quality_scores:
        logger.debug(f"No scores above threshold {MIN_THRESHOLD}")
        return 0.0
    
    # Calculate base confidence (average of high-quality scores)
    base_confidence = sum(high_quality_scores) / len(high_quality_scores)
    
    # Quality ratio: proportion of chunks above threshold
    quality_ratio = len(high_quality_scores) / len(scores)
    
    # Final confidence: balance score quality with quantity
    # Square root dampens the penalty for few chunks
    confidence = base_confidence * (quality_ratio ** 0.5)
    
    # Ensure in [0, 1] range
    confidence = max(0.0, min(1.0, confidence))
    
    logger.debug(
        f"Confidence: {confidence:.4f} "
        f"(base={base_confidence:.4f}, "
        f"quality={len(high_quality_scores)}/{len(scores)}, "
        f"ratio={quality_ratio:.4f})"
    )
    
    return confidence


def compute_extractive_confidence(max_similarity: float, num_sentences: int) -> float:
    """
    Compute confidence score for extractive QA answers.
    
    Based on:
    - Maximum sentence similarity score
    - Number of supporting sentences
    
    Args:
        max_similarity: Highest sentence-question similarity (0-1)
        num_sentences: Number of sentences in extracted answer
        
    Returns:
        Confidence score (0.0 to 1.0)
        
    Notes:
        - High similarity (> 0.7) → High confidence (0.7-0.9)
        - Medium similarity (0.5-0.7) → Medium confidence (0.5-0.7)
        - Low similarity (< 0.5) → Low confidence (< 0.5)
        - More sentences → slightly higher confidence (up to +0.1 boost)
    """
    if max_similarity <= 0:
        return 0.0
    
    # Base confidence from max similarity
    base_confidence = max_similarity
    
    # Sentence count boost (max +0.1)
    # More supporting sentences = slightly more confidence
    sentence_boost = min(0.1, (num_sentences - MIN_SENTENCES_PER_ANSWER) * 0.02)
    
    confidence = min(0.95, base_confidence + sentence_boost)
    
    logger.debug(
        f"Extractive confidence: {confidence:.4f} "
        f"(base={base_confidence:.4f}, "
        f"sentences={num_sentences}, "
        f"boost={sentence_boost:.4f})"
    )
    
    return confidence


def get_model_info() -> dict:
    """
    Get information about loaded T5 model.
    
    Returns:
        Dictionary with model info
    """
    if _qa_model is None:
        return {
            "model_loaded": False,
            "model_name": QA_MODEL_NAME,
            "device": DEVICE
        }
    
    return {
        "model_loaded": True,
        "model_name": QA_MODEL_NAME,
        "device": DEVICE,
        "max_input_length": MAX_INPUT_LENGTH,
        "max_output_length": MAX_OUTPUT_LENGTH,
        "parameters": sum(p.numel() for p in _qa_model.parameters())
    }
