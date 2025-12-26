# backend/app/services/websearch.py

"""
 Web Search Fallback using DuckDuckGo.

Provides safe web search when document context has low confidence:
- DuckDuckGo search (no API key needed)
- Top 5 results extraction
- Clean text formatting
- Deduplication
- Fail-safe error handling
"""

import logging
from typing import List, Dict
import re

logger = logging.getLogger(__name__)

# Web search configuration
MAX_WEB_RESULTS = 5
MIN_SNIPPET_LENGTH = 20  # Minimum snippet length to keep


def search_web(query: str) -> List[Dict[str, str]]:
    """
    Search web using DuckDuckGo and return top results.
    
    Returns structured results with title, snippet, and source.
    
    Args:
        query: Search query string
        
    Returns:
        List of result dictionaries with keys:
        - title: str
        - snippet: str
        - source: str
        
    Raises:
        ValueError: If query is empty
        RuntimeError: If search fails
    """
    # Validate query
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty")
    
    query = query.strip()
    
    logger.info(f"Web search query: {query}")
    
    try:
        # Import DuckDuckGo search
        from duckduckgo_search import DDGS
        
        # Perform search
        with DDGS() as ddgs:
            # Get search results (iterator)
            results_raw = list(ddgs.text(
                query,
                max_results=MAX_WEB_RESULTS
            ))
        
        logger.debug(f"Found {len(results_raw)} raw results")
        
        # Process and clean results
        processed_results = []
        seen_urls = set()  # For deduplication
        
        for result in results_raw:
            # Extract fields
            title = result.get("title", "").strip()
            snippet = result.get("body", "").strip()
            url = result.get("href", "").strip()
            
            # Skip if missing critical fields
            if not title or not snippet or not url:
                logger.debug("Skipping result with missing fields")
                continue
            
            # Skip if snippet too short
            if len(snippet) < MIN_SNIPPET_LENGTH:
                logger.debug(f"Skipping short snippet: {len(snippet)} chars")
                continue
            
            # Deduplicate by URL
            if url in seen_urls:
                logger.debug(f"Skipping duplicate URL: {url}")
                continue
            
            seen_urls.add(url)
            
            # Clean text
            title_clean = clean_text(title)
            snippet_clean = clean_text(snippet)
            
            # Extract domain for source
            source = extract_domain(url)
            
            processed_results.append({
                "title": title_clean,
                "snippet": snippet_clean,
                "source": source
            })
        
        logger.info(f"Processed {len(processed_results)} web results")
        
        return processed_results
        
    except ImportError:
        logger.error("duckduckgo_search package not installed")
        raise RuntimeError(
            "Web search requires duckduckgo-search package. "
            "Install with: pip install duckduckgo-search"
        )
    
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        raise RuntimeError(f"Web search failed: {str(e)}")


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and special characters.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_domain(url: str) -> str:
    """
    Extract domain from URL for source attribution.
    
    Args:
        url: Full URL string
        
    Returns:
        Domain name (e.g., "wikipedia.org")
    """
    if not url:
        return "unknown"
    
    try:
        # Remove protocol
        domain = re.sub(r'^https?://', '', url)
        
        # Remove www
        domain = re.sub(r'^www\.', '', domain)
        
        # Take first part (domain)
        domain = domain.split('/')[0]
        
        return domain
        
    except Exception:
        return "unknown"


def merge_web_context(results: List[Dict[str, str]]) -> str:
    """
    Merge web search results into single context string.
    
    Format: Each result as "Title: Snippet [Source]"
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Merged context string
    """
    if not results:
        return "No web results found."
    
    context_parts = []
    
    for i, result in enumerate(results, 1):
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        source = result.get("source", "unknown")
        
        # Format: [1] Title: Snippet [source]
        part = f"[{i}] {title}: {snippet} [{source}]"
        context_parts.append(part)
    
    context = "\n\n".join(context_parts)
    
    logger.debug(f"Merged web context: {len(context)} chars from {len(results)} results")
    
    return context


def get_web_context_for_question(question: str) -> tuple[str, List[Dict[str, str]]]:
    """
    Get web context for a question.
    
    Performs web search and returns both merged context and raw results.
    
    Args:
        question: User's question
        
    Returns:
        Tuple of (context_string, results_list)
        
    Raises:
        ValueError: If question is empty
        RuntimeError: If web search fails
    """
    # Validate question
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    # Search web
    results = search_web(question)
    
    # Merge into context
    context = merge_web_context(results)
    
    return context, results
