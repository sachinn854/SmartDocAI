# backend/app/services/summarizer.py

"""
AI-based document summarization service using T5.

Design principles:
- Singleton model loading (loaded once, reused across requests)
- Chunk-based hierarchical summarization (MAP-REDUCE approach)
- Noise cleaning (emails, phones, headers, page numbers)
- Deduplication (removes repetitive content)
- Caching via summaries.json (avoid recomputation)
- Model-agnostic design (easy to switch T5-small → T5-base → PEGASUS)

Scalability notes:
- Model is loaded lazily on first request
- Heavy computation isolated in service layer
- Thread-safe for concurrent users
- Summaries cached on disk, not in database (faster reads)
"""

import json
import re
from pathlib import Path
from typing import Optional, Dict, List
from threading import Lock

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from app.core.config import get_settings


# Global singleton for model and tokenizer
_model = None
_tokenizer = None
_model_lock = Lock()

settings = get_settings()


def clean_noise_from_text(text: str) -> str:
    """
    Remove metadata noise from text before summarization.
    
    Removes:
    - Email addresses
    - Phone numbers (various formats)
    - URLs
    - Repeated headers/footers
    - Page numbers
    - Date patterns
    - Excessive whitespace
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text with noise removed
    """
    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove phone numbers (multiple formats)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    text = re.sub(r'\(\d{3}\)\s?\d{3}[-.]?\d{4}', '', text)
    text = re.sub(r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),])+', '', text)
    
    # Remove page numbers (Page 1, Page 2, etc.)
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+\s+of\s+\d+\b', '', text, flags=re.IGNORECASE)
    
    # Remove date patterns (MM/DD/YYYY, DD-MM-YYYY, etc.)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)
    
    # Remove repeated short lines (headers/footers that appear on every page)
    lines = text.split('\n')
    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 50 and len(stripped) > 5:  # Short lines only
            line_counts[stripped] = line_counts.get(stripped, 0) + 1
    
    # Remove lines that appear more than 3 times (likely headers/footers)
    repeated_lines = {line for line, count in line_counts.items() if count > 3}
    lines = [line for line in lines if line.strip() not in repeated_lines]
    text = '\n'.join(lines)
    
    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple blank lines → double
    text = re.sub(r' +', ' ', text)  # Multiple spaces → single
    text = text.strip()
    
    return text


def deduplicate_sentences(text: str) -> str:
    """
    Remove duplicate or near-duplicate sentences from text.
    
    Uses simple string-based matching to detect repetition.
    Preserves order of first occurrence.
    
    Args:
        text: Text with potential duplicates
        
    Returns:
        Deduplicated text
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    seen = set()
    unique_sentences = []
    
    for sentence in sentences:
        # Normalize for comparison (lowercase, remove extra spaces)
        normalized = ' '.join(sentence.lower().split())
        
        if normalized not in seen and len(normalized) > 10:  # Skip very short sentences
            seen.add(normalized)
            unique_sentences.append(sentence)
    
    return '. '.join(unique_sentences) + '.'


def deduplicate_list(items: List[str]) -> List[str]:
    """
    Remove duplicate items from a list while preserving order.
    
    Args:
        items: List of strings (e.g., bullet points)
        
    Returns:
        Deduplicated list
    """
    seen = set()
    unique_items = []
    
    for item in items:
        # Normalize for comparison
        normalized = ' '.join(item.lower().split())
        
        if normalized not in seen:
            seen.add(normalized)
            unique_items.append(item)
    
    return unique_items


def load_model():
    """
    Load T5 model and tokenizer (singleton pattern).
    
    Thread-safe lazy loading ensures model is loaded only once
    across all requests, improving performance and memory efficiency.
    
    Returns:
        tuple: (tokenizer, model)
    """
    global _model, _tokenizer
    
    # Thread-safe check
    with _model_lock:
        if _model is None or _tokenizer is None:
            print(f"🔄 Loading summarization model: {settings.SUMMARIZER_MODEL}")
            
            _tokenizer = AutoTokenizer.from_pretrained(
                settings.SUMMARIZER_MODEL,
                model_max_length=512
            )
            _model = AutoModelForSeq2SeqLM.from_pretrained(
                settings.SUMMARIZER_MODEL
            )
            
            print(f"✅ Model loaded successfully")
        
        return _tokenizer, _model


def chunk_text(text: str, max_words: int = 150, overlap_words: int = 25) -> List[str]:
    """
    Split long text into overlapping chunks respecting paragraph boundaries.
    
    Improved chunking strategy:
    - Optimal chunk size: 120-180 words (prevents semantic collapse)
    - Maximum chunk size: 200 words
    - Minimum chunk size: 80 words
    - Overlap: 20-30 words (ensures context continuity without redundancy)
    - Respects paragraph boundaries when possible
    - Prevents splitting mid-sentence
    
    Args:
        text: Input text to chunk
        max_words: Maximum words per chunk (default 150)
        overlap_words: Words to overlap between chunks (default 25)
        
    Returns:
        List of text chunks
    """
    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for para in paragraphs:
        para_words = para.split()
        para_word_count = len(para_words)
        
        # If single paragraph is too long (>200 words), split it into sentences
        if para_word_count > 200:
            # Split by sentences and treat each as mini-paragraph
            sentences = para.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                sent_word_count = len(sent.split())
                
                if current_word_count + sent_word_count > max_words and current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(chunk_text)
                    
                    overlap_text = ' '.join(chunk_text.split()[-overlap_words:])
                    current_chunk = [overlap_text, sent]
                    current_word_count = len(overlap_text.split()) + sent_word_count
                else:
                    current_chunk.append(sent)
                    current_word_count += sent_word_count
        # Normal paragraph handling
        elif current_word_count + para_word_count > max_words and current_chunk:
            # Save current chunk
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap from previous
            overlap_text = ' '.join(chunk_text.split()[-overlap_words:])
            current_chunk = [overlap_text, para]
            current_word_count = len(overlap_text.split()) + para_word_count
        else:
            # Add paragraph to current chunk
            current_chunk.append(para)
            current_word_count += para_word_count
    
    # Add final chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    # Fallback: if no chunks created (no paragraphs), use word-based chunking
    if not chunks:
        words = text.split()
        if len(words) <= max_words:
            return [text]
        
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            start += (max_words - overlap_words)
    
    return chunks


def summarize_text(
    text: str,
    max_length: int,
    min_length: int,
    tokenizer,
    model,
    instruction: str = None
) -> str:
    """
    Generate summary for a single text chunk using T5 or BART.
    
    Automatically detects model type and adjusts input format:
    - T5 models: Require "summarize:" prefix
    - BART models: No prefix needed
    
    Args:
        text: Text to summarize
        max_length: Maximum summary length in tokens
        min_length: Minimum summary length in tokens
        tokenizer: T5 or BART tokenizer
        model: T5 or BART model
        instruction: Optional specialized instruction for summary style
        
    Returns:
        Generated summary
    """
    # Detect model type from config
    model_name = settings.SUMMARIZER_MODEL.lower()
    is_t5 = "t5" in model_name
    is_bart = "bart" in model_name
    
    # Prepare input based on model type
    if is_t5:
        # T5 requires "summarize:" prefix
        if instruction:
            input_text = f"summarize: {instruction} {text}"
        else:
            input_text = f"summarize: {text}"
    elif is_bart:
        # BART doesn't need prefix (use instruction if provided)
        if instruction:
            input_text = f"{instruction} {text}"
        else:
            input_text = text
    else:
        # Default to T5 format for unknown models
        if instruction:
            input_text = f"summarize: {instruction} {text}"
        else:
            input_text = f"summarize: {text}"
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    
    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    # Decode
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary


def summarize_chunks(chunks: List[str], max_length: int, min_length: int, instruction: str = None) -> str:
    """
    Summarize multiple chunks and merge results.
    
    Strategy:
    1. Summarize each chunk independently
    2. Concatenate chunk summaries
    3. If combined length is still long, summarize again
    
    This two-pass approach handles very long documents effectively.
    
    Args:
        chunks: List of text chunks
        max_length: Target max length for final summary
        min_length: Target min length for final summary
        instruction: Optional specialized instruction for summary style
        
    Returns:
        Merged summary
    """
    tokenizer, model = load_model()
    
    # Summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        summary = summarize_text(
            chunk,
            max_length=150,  # Intermediate summary length
            min_length=30,
            tokenizer=tokenizer,
            model=model,
            instruction=instruction
        )
        chunk_summaries.append(summary)
    
    # If single chunk, return its summary
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]
    
    # Merge chunk summaries
    merged_text = ' '.join(chunk_summaries)
    
    # If merged text is still long, summarize again
    merged_words = merged_text.split()
    if len(merged_words) > 300:
        final_summary = summarize_text(
            merged_text,
            max_length=max_length,
            min_length=min_length,
            tokenizer=tokenizer,
            model=model,
            instruction=instruction
        )
        return final_summary
    
    return merged_text


def summarize_single_chunk(chunk: str, tokenizer, model) -> str:
    """
    Summarize a single chunk into 3-4 detailed sentences (MAP step).
    
    This is the foundation of chunk-wise hierarchical summarization.
    Each chunk is summarized independently to preserve technical details.
    These summaries become paragraphs in the detailed summary.
    
    IMPORTANT: Generate RICH summaries with technical depth, not generic overviews.
    
    Args:
        chunk: Text chunk to summarize
        tokenizer: T5 tokenizer
        model: T5 model
        
    Returns:
        Chunk summary (3-4 sentences with technical details preserved)
    """
    instruction = "Summarize this section in 3-4 clear sentences. Preserve important technical details, concepts, and specifics. Avoid generic statements."
    
    summary = summarize_text(
        chunk,
        max_length=120,  # Increased from 100 to allow more detail
        min_length=50,   # Increased from 40 to ensure substance
        tokenizer=tokenizer,
        model=model,
        instruction=instruction
    )
    
    return summary.strip()


def generate_chunk_summaries(text: str) -> List[str]:
    """
    Generate summaries for each chunk of text (MAP step).
    
    This is step 1 of hierarchical chunk-based summarization.
    Each chunk is summarized independently to preserve details and prevent repetition.
    These summaries become the foundation for all summary types.
    
    Args:
        text: Full cleaned text
        
    Returns:
        List of chunk summaries (one per chunk, 3-4 sentences each with technical depth)
    """
    tokenizer, model = load_model()
    
    # Chunk with improved parameters
    chunks = chunk_text(text, max_words=150, overlap_words=25)
    
    # Limit to first 15 chunks for performance (covers ~1800-2000 words)
    chunks = chunks[:15]
    
    print(f"   📦 Processing {len(chunks)} chunks (MAP step)")
    
    # Summarize each chunk independently
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        chunk_summary = summarize_single_chunk(chunk, tokenizer, model)
        
        # Only add if meaningful (not too short) - lowered threshold to preserve more
        if chunk_summary and len(chunk_summary.split()) >= 8:  # Lowered from 10 to 8
            chunk_summaries.append(chunk_summary)
            print(f"   ✓ Chunk {i+1}/{len(chunks)}: {len(chunk_summary.split())} words")
        else:
            print(f"   ⚠ Chunk {i+1}/{len(chunks)}: Skipped (too short: {len(chunk_summary.split()) if chunk_summary else 0} words)")
    
    return chunk_summaries


def generate_short_summary(text: str, chunk_summaries: List[str] = None) -> str:
    """
    Generate concise abstract (3-4 sentences) - REDUCE step.
    
    Hierarchical compression:
    Chunk summaries → Medium summary → Short summary (abstract)
    
    Args:
        text: Cleaned text (used if chunk_summaries not provided)
        chunk_summaries: Pre-computed chunk summaries (optimal)
        
    Returns:
        Short summary (3-4 sentences, very high-level overview)
    """
    tokenizer, model = load_model()
    
    # If chunk summaries provided, use hierarchical approach
    if chunk_summaries:
        # First build medium summary from chunks
        combined_text = " ".join(chunk_summaries)
        
        instruction = "Combine these key points into a clear, cohesive summary. Remove any repetition."
        
        medium = summarize_text(
            combined_text,
            max_length=180,
            min_length=100,
            tokenizer=tokenizer,
            model=model,
            instruction=instruction
        )
        
        # Then compress to short abstract
        instruction = "Write a very concise abstract (3-4 sentences) capturing only the central theme and main conclusion."
        
        short = summarize_text(
            medium,
            max_length=90,
            min_length=50,
            tokenizer=tokenizer,
            model=model,
            instruction=instruction
        )
        
        # Deduplicate sentences
        short = deduplicate_sentences(short)
        
        return short.strip()
    
    # Fallback: direct summarization for short documents
    chunks = chunk_text(text, max_words=150, overlap_words=25)
    
    instruction = "Write a very concise abstract (3-4 sentences) capturing only the central theme and main conclusion."
    
    summary = summarize_chunks(
        chunks,
        max_length=90,
        min_length=50,
        instruction=instruction
    )
    
    summary = deduplicate_sentences(summary)
    
    return summary.strip()


def generate_medium_summary(text: str, chunk_summaries: List[str] = None) -> str:
    """
    Generate medium-length summary (6-8 sentences) - REDUCE step.
    
    Hierarchical approach:
    Chunk summaries → Merged and compressed → Medium summary
    
    Args:
        text: Cleaned text (used if chunk_summaries not provided)
        chunk_summaries: Pre-computed chunk summaries (optimal)
        
    Returns:
        Medium summary (6-8 sentences, concise explanation)
    """
    tokenizer, model = load_model()
    
    # If chunk summaries provided, merge and compress them
    if chunk_summaries:
        combined_text = " ".join(chunk_summaries)
        
        instruction = "Combine these key points into a clear, cohesive summary. Remove any repetition. Provide a complete explanation in 6-8 sentences."
        
        summary = summarize_text(
            combined_text,
            max_length=180,
            min_length=100,
            tokenizer=tokenizer,
            model=model,
            instruction=instruction
        )
        
        # Deduplicate sentences
        summary = deduplicate_sentences(summary)
        
        return summary.strip()
    
    # Fallback: direct summarization for short documents
    chunks = chunk_text(text, max_words=150, overlap_words=25)
    
    instruction = "Combine these key points into a clear, cohesive summary. Remove any repetition. Provide a complete explanation in 6-8 sentences."
    
    summary = summarize_chunks(
        chunks,
        max_length=180,
        min_length=100,
        instruction=instruction
    )
    
    summary = deduplicate_sentences(summary)
    
    return summary.strip()


def generate_detailed_summary(text: str, chunk_summaries: List[str] = None) -> List[str]:
    """
    Generate structured detailed summary (MULTIPLE paragraphs/bullets).
    
    CRITICAL: Do NOT re-summarize chunk summaries.
    Each chunk summary = one paragraph in detailed summary.
    Only apply light deduplication (exact duplicates only).
    
    This preserves chunk-level richness and technical details.
    
    Args:
        text: Cleaned text (used if chunk_summaries not provided)
        chunk_summaries: Pre-computed chunk summaries (optimal)
        
    Returns:
        List of detailed paragraphs (one per chunk, preserving richness)
    """
    # If chunk summaries provided, use them directly WITHOUT further compression
    if chunk_summaries:
        # Apply LIGHT deduplication only - remove exact duplicates, keep semantic variations
        seen = set()
        unique_summaries = []
        
        for summary in chunk_summaries:
            # Normalize for comparison (lowercase, strip whitespace)
            normalized = summary.lower().strip()
            
            # Only skip if EXACT duplicate (not semantic similarity)
            if normalized not in seen:
                seen.add(normalized)
                unique_summaries.append(summary)
        
        # Cap at 12 paragraphs for readability (increased from before)
        # Each paragraph is a full chunk summary (3-4 sentences)
        return unique_summaries[:12]
    
    # Fallback: generate chunk summaries now
    tokenizer, model = load_model()
    
    # Create chunks with improved parameters
    chunks = chunk_text(text, max_words=150, overlap_words=25)
    chunks = chunks[:10]
    
    # Summarize each chunk
    instruction = "Extract only the key ideas from this section in 3-4 clear sentences. Ignore headers, names, dates, and repeated information."
    
    bullet_points = []
    
    for chunk in chunks:
        summary = summarize_text(
            chunk,
            max_length=100,
            min_length=40,
            tokenizer=tokenizer,
            model=model,
            instruction=instruction
        )
        
        summary = summary.strip()
        if summary and len(summary.split()) >= 10:
            bullet_points.append(summary)
    
    # Deduplicate
    bullet_points = deduplicate_list(bullet_points)
    
    return bullet_points[:12]


def generate_all_summaries(cleaned_text: str) -> Dict[str, any]:
    """
    Generate all three summary types using improved hierarchical approach.
    
    This is the main entry point for summarization.
    
    Workflow:
    1. Clean noise from text (emails, phones, headers, page numbers)
    2. Check document length (short vs long handling)
    3. MAP step: Generate chunk summaries independently
    4. REDUCE step: Build short/medium from chunks
    5. Detailed: Use deduplicated chunk summaries
    6. Apply deduplication to all outputs
    
    API Contract (UNCHANGED):
    Returns dictionary with short_summary, medium_summary, detailed_summary
    
    Args:
        cleaned_text: Pre-cleaned text from cleaned.txt
        
    Returns:
        Dictionary with short_summary, medium_summary, detailed_summary
    """
    print(f"📝 Generating summaries for text ({len(cleaned_text)} chars)")
    
    # STEP 1: Clean noise from text
    print(f"   🧹 Cleaning noise (emails, phones, headers, page numbers)...")
    cleaned_text = clean_noise_from_text(cleaned_text)
    print(f"   ✓ Cleaned: {len(cleaned_text)} chars remaining")
    
    # STEP 2: Check word count - short documents get single summary
    word_count = len(cleaned_text.split())
    print(f"📊 Document word count: {word_count}")
    
    if word_count < 300:
        # Document too short for hierarchical summarization
        print(f"⚠️  Document < 300 words - generating single summary for all levels")
        single_summary = generate_short_summary(cleaned_text, chunk_summaries=None)
        
        summaries = {
            "short_summary": single_summary,
            "medium_summary": single_summary,
            "detailed_summary": [single_summary]  # Convert to list for consistency
        }
        
        print(f"✅ Single summary generated: {len(single_summary.split())} words")
        return summaries
    
    # STEP 3: MAP STEP - Generate chunk summaries (foundation)
    print(f"   🔄 Step 1 (MAP): Generating chunk summaries...")
    chunk_summaries = generate_chunk_summaries(cleaned_text)
    print(f"   ✓ Generated {len(chunk_summaries)} chunk summaries")
    
    # STEP 4: REDUCE STEP - Build detailed summary (deduplicated chunks)
    print(f"   🔄 Step 2 (REDUCE): Building detailed summary...")
    detailed = generate_detailed_summary(cleaned_text, chunk_summaries=chunk_summaries)
    print(f"   ✓ Detailed: {len(detailed)} bullet points")
    
    # STEP 5: REDUCE STEP - Build medium summary (compressed merge)
    print(f"   🔄 Step 3 (REDUCE): Building medium summary...")
    medium = generate_medium_summary(cleaned_text, chunk_summaries=chunk_summaries)
    print(f"   ✓ Medium: {len(medium.split())} words")
    
    # STEP 6: REDUCE STEP - Build short summary (abstract of medium)
    print(f"   🔄 Step 4 (REDUCE): Building short summary...")
    short = generate_short_summary(cleaned_text, chunk_summaries=chunk_summaries)
    print(f"   ✓ Short: {len(short.split())} words")
    
    # STEP 7: Final verification - ensure quality constraints
    short_words = set(short.lower().split())
    medium_words = set(medium.lower().split())
    
    # Calculate word counts for detailed summary (sum of all paragraphs)
    detailed_word_count = sum(len(para.split()) for para in detailed)
    medium_word_count = len(medium.split())
    short_word_count = len(short.split())
    
    overlap_short_medium = len(short_words & medium_words) / max(len(short_words), 1)
    
    print(f"   📊 Summary statistics:")
    print(f"      - Detailed: {detailed_word_count} total words across {len(detailed)} paragraphs")
    print(f"      - Medium: {medium_word_count} words")
    print(f"      - Short: {short_word_count} words")
    print(f"      - Short-Medium overlap: {overlap_short_medium*100:.1f}%")
    
    # CRITICAL CHECK: Detailed MUST be longer than medium
    if detailed_word_count <= medium_word_count:
        print(f"   ⚠️  WARNING: Detailed summary ({detailed_word_count} words) is not longer than medium ({medium_word_count} words)")
        print(f"              This indicates chunk-level richness may have been lost")
    
    summaries = {
        "short_summary": short,
        "medium_summary": medium,
        "detailed_summary": detailed
    }
    
    print(f"✅ All summaries generated successfully")
    print(f"   - Short: {short_word_count} words")
    print(f"   - Medium: {medium_word_count} words")
    print(f"   - Detailed: {detailed_word_count} words across {len(detailed)} paragraphs")
    
    return summaries


def save_summaries(user_id: int, doc_id: int, summaries: Dict[str, any]) -> Path:
    """
    Save summaries to summaries.json.
    
    Caching strategy:
    - Summaries saved to disk (not database) for fast reads
    - JSON format allows easy updates and extensions
    - File-based caching scales better than DB for large text
    
    Args:
        user_id: User ID
        doc_id: Document ID
        summaries: Dictionary of summaries
        
    Returns:
        Path to saved summaries.json
    """
    summaries_path = settings.UPLOAD_DIR / str(user_id) / str(doc_id) / "summaries.json"
    
    # Write summaries as JSON
    with summaries_path.open('w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Summaries saved to: {summaries_path}")
    
    return summaries_path


def load_summaries(user_id: int, doc_id: int) -> Optional[Dict[str, any]]:
    """
    Load cached summaries from summaries.json.
    
    Returns None if file doesn't exist (summaries not yet generated).
    
    Args:
        user_id: User ID
        doc_id: Document ID
        
    Returns:
        Dictionary of summaries or None if not found
    """
    summaries_path = settings.UPLOAD_DIR / str(user_id) / str(doc_id) / "summaries.json"
    
    if not summaries_path.exists():
        return None
    
    try:
        with summaries_path.open('r', encoding='utf-8') as f:
            summaries = json.load(f)
        
        print(f"📂 Loaded cached summaries from: {summaries_path}")
        return summaries
    
    except Exception as e:
        print(f"⚠️ Error loading summaries: {e}")
        return None
