# backend/app/services/pipeline_diagnostics.py

"""
Pipeline diagnostics for Step 6A: Embedding + Indexing pipeline.

This module provides diagnostic utilities to verify the state of the
document processing pipeline WITHOUT modifying existing logic.

Purpose:
- Verify each stage of the pipeline (cleaned text → chunks → embeddings → FAISS)
- Provide clear status reports for debugging
- Guard against silent failures
- Help users understand why /ask fails

DO NOT:
- Change chunking logic
- Change embedding logic
- Change indexing logic
- Auto-fix issues
- Retry silently
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

from app.core.config import get_settings
from app.services.retriever import is_document_indexed, get_index_stats

# Configure logging
logger = logging.getLogger(__name__)

settings = get_settings()


def check_document_pipeline_status(user_id: int, doc_id: int) -> Dict[str, Any]:
    """
    Comprehensive diagnostic check for document pipeline status.
    
    Verifies and logs the state of each pipeline stage:
    1. cleaned.txt exists and is non-empty
    2. Can read cleaned text successfully
    3. Would generate chunks > 0 (without actually generating)
    4. FAISS index exists
    5. FAISS index has vectors (ntotal > 0)
    
    This is a READ-ONLY diagnostic function - it does not modify anything.
    
    Args:
        user_id: User ID
        doc_id: Document ID
        
    Returns:
        Status dictionary with keys:
        - cleaned_text_exists: bool
        - cleaned_text_size_bytes: int
        - cleaned_word_count: int
        - estimated_chunks: int (rough estimate)
        - faiss_index_exists: bool
        - faiss_vectors_count: int
        - ready_for_qa: bool (overall status)
        - issues: List[str] (problems found)
        - warnings: List[str] (potential issues)
        
    Example:
        >>> status = check_document_pipeline_status(user_id=1, doc_id=42)
        >>> if status['ready_for_qa']:
        ...     print("Document ready for Q&A")
        >>> else:
        ...     print(f"Issues: {status['issues']}")
    """
    logger.info(f"🔍 Pipeline diagnostic: user_id={user_id}, doc_id={doc_id}")
    
    status = {
        "user_id": user_id,
        "doc_id": doc_id,
        "cleaned_text_exists": False,
        "cleaned_text_size_bytes": 0,
        "cleaned_word_count": 0,
        "estimated_chunks": 0,
        "faiss_index_exists": False,
        "faiss_vectors_count": 0,
        "ready_for_qa": False,
        "issues": [],
        "warnings": []
    }
    
    # Stage 1: Check cleaned.txt
    cleaned_path = settings.UPLOAD_DIR / str(user_id) / str(doc_id) / "cleaned.txt"
    
    if not cleaned_path.exists():
        status["issues"].append(f"cleaned.txt not found at {cleaned_path}")
        logger.warning(f"❌ cleaned.txt missing: {cleaned_path}")
        return status
    
    status["cleaned_text_exists"] = True
    
    # Check file size
    try:
        file_size = cleaned_path.stat().st_size
        status["cleaned_text_size_bytes"] = file_size
        
        if file_size == 0:
            status["issues"].append("cleaned.txt is empty (0 bytes)")
            logger.warning(f"❌ cleaned.txt is empty: {cleaned_path}")
            return status
        
        logger.info(f"✅ cleaned.txt exists: {file_size} bytes")
        
    except Exception as e:
        status["issues"].append(f"Cannot read cleaned.txt: {str(e)}")
        logger.error(f"❌ Error reading cleaned.txt: {e}")
        return status
    
    # Stage 2: Read and analyze text content
    try:
        text_content = cleaned_path.read_text(encoding="utf-8")
        word_count = len(text_content.split())
        status["cleaned_word_count"] = word_count
        
        if word_count == 0:
            status["issues"].append("cleaned.txt has no words")
            logger.warning(f"❌ cleaned.txt has no words")
            return status
        
        if word_count < 50:
            status["warnings"].append(f"Document is very short ({word_count} words). May not chunk properly.")
            logger.warning(f"⚠️ Short document: {word_count} words")
        
        # Rough estimate of chunks (without actually chunking)
        # Typical chunk size: 300 words with 80 word overlap
        # This is just an estimate for diagnostics
        estimated_chunks = max(1, (word_count - 80) // (300 - 80) + 1) if word_count > 80 else 1
        status["estimated_chunks"] = estimated_chunks
        
        logger.info(f"✅ Text content: {word_count} words, ~{estimated_chunks} chunks estimated")
        
    except Exception as e:
        status["issues"].append(f"Cannot parse cleaned.txt: {str(e)}")
        logger.error(f"❌ Error parsing cleaned.txt: {e}")
        return status
    
    # Stage 3: Check FAISS index
    index_exists = is_document_indexed(user_id, doc_id)
    status["faiss_index_exists"] = index_exists
    
    if not index_exists:
        status["issues"].append("FAISS index not found. Run POST /documents/embed/{doc_id} first.")
        logger.warning(f"❌ FAISS index missing for user {user_id}, doc {doc_id}")
        return status
    
    logger.info(f"✅ FAISS index exists")
    
    # Stage 4: Check index stats
    try:
        index_stats = get_index_stats(user_id, doc_id)
        vector_count = index_stats.get('num_vectors', 0)
        status["faiss_vectors_count"] = vector_count
        
        if vector_count == 0:
            status["issues"].append("FAISS index exists but has 0 vectors")
            logger.error(f"❌ FAISS index is empty: 0 vectors")
            return status
        
        logger.info(f"✅ FAISS index: {vector_count} vectors, dim={index_stats.get('dimension', '?')}")
        
    except Exception as e:
        status["issues"].append(f"Cannot read FAISS index stats: {str(e)}")
        logger.error(f"❌ Error reading index stats: {e}")
        return status
    
    # Final assessment
    if not status["issues"]:
        status["ready_for_qa"] = True
        logger.info(f"✅ Pipeline ready for Q&A")
    else:
        logger.warning(f"❌ Pipeline not ready: {len(status['issues'])} issues found")
    
    return status


def verify_pipeline_stage(
    stage_name: str,
    user_id: int,
    doc_id: int,
    raise_on_failure: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Verify a specific pipeline stage with clear error messages.
    
    Guard rails for explicit error reporting (NO auto-fix, NO retry).
    
    Args:
        stage_name: One of ['cleaned_text', 'chunks', 'embeddings', 'faiss_index']
        user_id: User ID
        doc_id: Document ID
        raise_on_failure: If True, raise descriptive exception on failure
        
    Returns:
        (success: bool, error_message: Optional[str])
        
    Raises:
        FileNotFoundError: If cleaned.txt or FAISS index missing (when raise_on_failure=True)
        ValueError: If stage validation fails (when raise_on_failure=True)
    """
    logger.info(f"🔍 Verifying stage '{stage_name}' for user {user_id}, doc {doc_id}")
    
    if stage_name == 'cleaned_text':
        cleaned_path = settings.UPLOAD_DIR / str(user_id) / str(doc_id) / "cleaned.txt"
        
        if not cleaned_path.exists():
            error_msg = (
                f"❌ PIPELINE FAILURE: cleaned.txt not found\n"
                f"   Path: {cleaned_path}\n"
                f"   User: {user_id}, Doc: {doc_id}\n"
                f"   Action: Upload document first using POST /documents/upload"
            )
            logger.error(error_msg)
            if raise_on_failure:
                raise FileNotFoundError(error_msg)
            return False, error_msg
        
        try:
            content = cleaned_path.read_text(encoding="utf-8")
            if not content or len(content.strip()) == 0:
                error_msg = (
                    f"❌ PIPELINE FAILURE: cleaned.txt is empty\n"
                    f"   Path: {cleaned_path}\n"
                    f"   User: {user_id}, Doc: {doc_id}\n"
                    f"   Action: Document upload may have failed. Re-upload document."
                )
                logger.error(error_msg)
                if raise_on_failure:
                    raise ValueError(error_msg)
                return False, error_msg
                
            logger.info(f"✅ cleaned.txt verified: {len(content)} chars")
            return True, None
            
        except Exception as e:
            error_msg = f"❌ PIPELINE FAILURE: Cannot read cleaned.txt: {str(e)}"
            logger.error(error_msg)
            if raise_on_failure:
                raise
            return False, error_msg
    
    elif stage_name == 'faiss_index':
        if not is_document_indexed(user_id, doc_id):
            index_path = settings.INDEX_DIR / str(user_id) / str(doc_id) / "faiss.index"
            error_msg = (
                f"❌ PIPELINE FAILURE: FAISS index not found\n"
                f"   Path: {index_path}\n"
                f"   User: {user_id}, Doc: {doc_id}\n"
                f"   Action: Index document first using POST /documents/embed/{doc_id}"
            )
            logger.error(error_msg)
            if raise_on_failure:
                raise FileNotFoundError(error_msg)
            return False, error_msg
        
        try:
            stats = get_index_stats(user_id, doc_id)
            vector_count = stats.get('num_vectors', 0)
            
            if vector_count == 0:
                error_msg = (
                    f"❌ PIPELINE FAILURE: FAISS index exists but has 0 vectors\n"
                    f"   User: {user_id}, Doc: {doc_id}\n"
                    f"   This indicates indexing failed silently.\n"
                    f"   Action: Re-run POST /documents/embed/{doc_id}"
                )
                logger.error(error_msg)
                if raise_on_failure:
                    raise ValueError(error_msg)
                return False, error_msg
            
            logger.info(f"✅ FAISS index verified: {vector_count} vectors")
            return True, None
            
        except Exception as e:
            error_msg = f"❌ PIPELINE FAILURE: Cannot read FAISS index: {str(e)}"
            logger.error(error_msg)
            if raise_on_failure:
                raise
            return False, error_msg
    
    else:
        error_msg = f"❌ Unknown stage: {stage_name}"
        logger.error(error_msg)
        if raise_on_failure:
            raise ValueError(error_msg)
        return False, error_msg


def format_pipeline_report(status: Dict[str, Any]) -> str:
    """
    Format pipeline status as readable report.
    
    Args:
        status: Status dict from check_document_pipeline_status()
        
    Returns:
        Formatted multi-line report string
    """
    report_lines = [
        "=" * 60,
        f"  PIPELINE DIAGNOSTIC REPORT",
        "=" * 60,
        f"User ID: {status['user_id']}",
        f"Doc ID: {status['doc_id']}",
        "",
        "📄 CLEANED TEXT:",
        f"  Exists: {'✅' if status['cleaned_text_exists'] else '❌'}",
        f"  Size: {status['cleaned_text_size_bytes']} bytes",
        f"  Words: {status['cleaned_word_count']}",
        f"  Est. Chunks: ~{status['estimated_chunks']}",
        "",
        "🔢 FAISS INDEX:",
        f"  Exists: {'✅' if status['faiss_index_exists'] else '❌'}",
        f"  Vectors: {status['faiss_vectors_count']}",
        "",
        "🎯 OVERALL STATUS:",
        f"  Ready for Q&A: {'✅ YES' if status['ready_for_qa'] else '❌ NO'}",
        "",
    ]
    
    if status['warnings']:
        report_lines.append("⚠️ WARNINGS:")
        for warning in status['warnings']:
            report_lines.append(f"  - {warning}")
        report_lines.append("")
    
    if status['issues']:
        report_lines.append("❌ ISSUES FOUND:")
        for issue in status['issues']:
            report_lines.append(f"  - {issue}")
        report_lines.append("")
    
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)
