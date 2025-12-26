# backend/app/services/extractor.py

"""
Text extraction service for various document formats.
Supports PDF, DOCX, TXT with OCR fallback for PDFs.
"""

from pathlib import Path
from typing import Optional

import pdfplumber
from docx import Document as DocxDocument
import pytesseract
from pdf2image import convert_from_path

from app.core.config import get_settings

settings = get_settings()

# Configure pytesseract path if specified
if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


def extract_pdf_text(pdf_path: Path) -> Optional[str]:
    """
    Extract text from PDF using pdfplumber.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text or None if no text found
    """
    try:
        text_content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
        
        combined_text = "\n".join(text_content)
        
        # Return None if text is empty or only whitespace
        return combined_text.strip() if combined_text.strip() else None
    
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None


def extract_docx_text(docx_path: Path) -> Optional[str]:
    """
    Extract text from DOCX file.
    
    Args:
        docx_path: Path to DOCX file
        
    Returns:
        Extracted text or None on error
    """
    try:
        doc = DocxDocument(docx_path)
        paragraphs = [paragraph.text for paragraph in doc.paragraphs]
        text = "\n".join(paragraphs)
        return text.strip() if text.strip() else None
    
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
        return None


def extract_txt_text(txt_path: Path) -> Optional[str]:
    """
    Extract text from plain text file.
    
    Args:
        txt_path: Path to TXT file
        
    Returns:
        File contents or None on error
    """
    try:
        text = txt_path.read_text(encoding="utf-8")
        return text.strip() if text.strip() else None
    
    except UnicodeDecodeError:
        # Try alternative encodings
        try:
            text = txt_path.read_text(encoding="latin-1")
            return text.strip() if text.strip() else None
        except Exception as e:
            print(f"Error reading text file: {e}")
            return None
    
    except Exception as e:
        print(f"Error extracting TXT text: {e}")
        return None


def apply_ocr(pdf_path: Path) -> Optional[str]:
    """
    Apply OCR to PDF using pytesseract.
    
    Converts PDF pages to images and extracts text using OCR.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        OCR-extracted text or None on error
    """
    try:
        # Convert PDF pages to images
        images = convert_from_path(pdf_path)
        
        ocr_text = []
        for i, image in enumerate(images):
            # Extract text from image using OCR
            text = pytesseract.image_to_string(image)
            if text.strip():
                ocr_text.append(text)
        
        combined_text = "\n".join(ocr_text)
        return combined_text.strip() if combined_text.strip() else None
    
    except Exception as e:
        print(f"Error applying OCR: {e}")
        return None


def extract_text_auto(file_path: Path) -> Optional[str]:
    """
    Automatically detect file type and extract text.
    
    For PDFs: tries pdfplumber first, falls back to OCR if no text found.
    
    Args:
        file_path: Path to document file
        
    Returns:
        Extracted raw text or None on failure
    """
    file_path = Path(file_path)
    file_ext = file_path.suffix.lower()
    
    if file_ext == ".pdf":
        # Try pdfplumber first
        text = extract_pdf_text(file_path)
        
        # If no text found, apply OCR
        if not text:
            print(f"No text found in PDF, applying OCR...")
            text = apply_ocr(file_path)
        
        return text
    
    elif file_ext == ".docx":
        return extract_docx_text(file_path)
    
    elif file_ext == ".txt":
        return extract_txt_text(file_path)
    
    else:
        print(f"Unsupported file extension: {file_ext}")
        return None
