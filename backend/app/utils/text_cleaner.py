# backend/app/utils/text_cleaner.py

"""
Professional-grade text cleaning utilities for NLP pipelines.
Optimized for summarization, embeddings, and RAG systems.
"""

import re
import unicodedata
from typing import Optional
from collections import Counter


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters and fix encoding issues.
    
    - Converts smart quotes to regular quotes
    - Normalizes accented characters
    - Removes control characters
    
    Args:
        text: Raw text with potential unicode issues
        
    Returns:
        Normalized text
    """
    # Normalize to NFKD form (compatibility decomposition)
    text = unicodedata.normalize('NFKD', text)
    
    # Remove control characters (except newline and tab)
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
    
    # Fix common unicode issues
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2013': '-',  # En dash
        '\u2014': '-',  # Em dash
        '\u2026': '...',  # Ellipsis
        '\xa0': ' ',    # Non-breaking space
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def remove_urls(text: str) -> str:
    """
    Remove all URLs from text.
    
    Matches:
    - http://example.com
    - https://example.com
    - www.example.com
    - example.com/path
    
    Args:
        text: Text containing URLs
        
    Returns:
        Text with URLs removed
    """
    # Remove HTTP(S) URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove www URLs
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove naked domains (basic pattern)
    text = re.sub(r'\b[a-zA-Z0-9-]+\.(com|org|net|edu|gov|io|co)\b', '', text)
    
    return text


def remove_html_tags(text: str) -> str:
    """
    Remove HTML/XML tags from text.
    
    Args:
        text: Text containing HTML tags
        
    Returns:
        Text with tags removed
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    return text


def remove_emojis(text: str) -> str:
    """
    Remove all emoji characters from text.
    
    Args:
        text: Text containing emojis
        
    Returns:
        Text with emojis removed
    """
    # Emoji unicode ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+",
        flags=re.UNICODE
    )
    
    text = emoji_pattern.sub('', text)
    
    return text


def remove_special_chars(text: str) -> str:
    """
    Remove special characters while preserving meaningful punctuation.
    
    Keeps: . , ? ! - ( ) ' "
    Removes: @ # $ % ^ & * _ + = < > ~ | { } [ ]
    
    Args:
        text: Text with special characters
        
    Returns:
        Text with noise characters removed
    """
    # Remove specific special characters, preserve meaningful ones
    text = re.sub(r'[@#$%^&*_+=<>~|{}\[\]\\]', '', text)
    
    return text


def remove_date_patterns(text: str) -> str:
    """
    Remove date patterns commonly found in lecture slides and documents.
    
    Matches patterns like:
    - 25 september 2023
    - 1 january 2024
    - 15 dec 2022
    
    Args:
        text: Text containing date patterns
        
    Returns:
        Text with date patterns removed
    """
    # Full month names
    date_pattern = r'\b\d{1,2}\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b'
    text = re.sub(date_pattern, '', text, flags=re.IGNORECASE)
    
    # Abbreviated month names
    date_pattern_short = r'\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{2,4}\b'
    text = re.sub(date_pattern_short, '', text, flags=re.IGNORECASE)
    
    # Common date formats: DD/MM/YYYY, MM/DD/YYYY, YYYY-MM-DD
    date_pattern_slash = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
    text = re.sub(date_pattern_slash, '', text)
    
    date_pattern_iso = r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
    text = re.sub(date_pattern_iso, '', text)
    
    return text


def remove_slide_numbers(text: str) -> str:
    """
    Remove slide number patterns like 'ethernet 88', 'lecture 110'.
    
    Only removes if pattern appears frequently (metadata, not content).
    
    Args:
        text: Text containing slide numbers
        
    Returns:
        Text with slide numbers removed
    """
    # Pattern: word followed by 2-4 digit number
    # Example: ethernet 88, lecture 110, slide 25
    slide_pattern = r'\b[a-zA-Z]+\s+\d{2,4}\b'
    
    # Find all matches
    matches = re.findall(slide_pattern, text, flags=re.IGNORECASE)
    
    # Count occurrences
    match_counts = Counter(m.lower() for m in matches)
    
    # Remove only frequent patterns (likely metadata)
    for pattern, count in match_counts.items():
        if count > 2:  # Appears 3+ times = metadata
            # Remove all occurrences
            text = re.sub(re.escape(pattern), '', text, flags=re.IGNORECASE)
    
    return text


def remove_repeated_lines(text: str) -> str:
    """
    Remove frequently repeated short lines (headers, footers, metadata).
    
    Strategy:
    - Lines appearing >3 times AND ≤6 words = metadata
    - Preserves real content (long sentences, unique lines)
    
    Examples removed:
    - Instructor names (repeated on every slide)
    - Course codes (repeated headers)
    - Page numbers
    - Copyright notices
    
    Args:
        text: Text with potential repeated metadata
        
    Returns:
        Text with repeated metadata removed
    """
    lines = text.splitlines()
    
    # Count line occurrences (case-insensitive, stripped)
    line_keys = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 0:
            line_keys.append(stripped.lower())
        else:
            line_keys.append('')  # Preserve empty lines
    
    counts = Counter(key for key in line_keys if key)
    
    # Filter lines
    cleaned_lines = []
    for i, line in enumerate(lines):
        key = line_keys[i]
        
        # Keep empty lines
        if not key:
            cleaned_lines.append(line)
            continue
        
        # Remove if: repeated >3 times AND short (≤6 words)
        word_count = len(key.split())
        if counts[key] > 3 and word_count <= 6:
            continue  # Skip this line (metadata)
        
        # Keep everything else
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def remove_metadata_noise(text: str) -> str:
    """
    Remove lecture slide metadata before summarization.
    
    Removes:
    - Date patterns
    - Repeated instructor names
    - Slide numbers
    - Repeated headers/footers
    
    Preserves:
    - Academic content
    - Section titles (occur once)
    - Long sentences
    - Unique information
    
    Args:
        text: Raw text from PDF/document
        
    Returns:
        Text with metadata removed
    """
    # Step 1: Remove date patterns
    text = remove_date_patterns(text)
    
    # Step 2: Remove slide number patterns
    text = remove_slide_numbers(text)
    
    # Step 3: Remove frequently repeated short lines
    text = remove_repeated_lines(text)
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    - Multiple spaces → single space
    - Multiple newlines → single newline
    - Remove tabs
    
    Args:
        text: Text with irregular whitespace
        
    Returns:
        Text with normalized whitespace
    """
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Multiple spaces to single space
    text = re.sub(r' +', ' ', text)
    
    # Multiple newlines to single newline
    text = re.sub(r'\n\s*\n+', '\n', text)
    
    return text


def clean_text(text: str) -> Optional[str]:
    """
    Apply full text cleaning pipeline.
    
    Pipeline order:
    1. Normalize unicode
    2. Remove URLs
    3. Remove HTML tags
    4. Remove emojis
    5. Remove special characters
    6. Remove metadata noise (dates, repeated lines, slide numbers)
    7. Normalize whitespace
    8. Convert to lowercase
    9. Strip and remove empty lines
    
    Optimized for:
    - Summarization models (T5/PEGASUS)
    - SentenceTransformer embeddings
    - FAISS retrieval
    - RAG + QA systems
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text ready for NLP processing, or None if empty
    """
    if not text or not text.strip():
        return None
    
    # Apply cleaning pipeline in order
    text = normalize_unicode(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_emojis(text)
    text = remove_special_chars(text)
    
    # CRITICAL: Remove metadata noise before summarization
    text = remove_metadata_noise(text)
    
    text = normalize_whitespace(text)
    
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Remove empty lines
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    text = '\n'.join(lines)
    
    # Return None if text is empty after cleaning
    return text if text else None
