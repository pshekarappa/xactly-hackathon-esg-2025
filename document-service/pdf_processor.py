import fitz  # PyMuPDF
import os
import re
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean extracted text by removing extra whitespace and other artifacts."""
    # Replace multiple spaces, tabs, and newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any non-printable characters
    text = re.sub(r'[^\x20-\x7E\s]', '', text)
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    logger.info(f"Processing PDF: {pdf_path}")
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        full_text = ""

        # Process each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text from the page
            page_text = page.get_text("text")
            full_text += page_text + "\n\n"
            
            # Advanced: extract text from images if needed
            # This would require OCR which is more complex
            # If needed, we can add pytesseract or other OCR tools
        
        # Clean the extracted text
        cleaned_text = clean_text(full_text)
        logger.info(f"Successfully extracted text from {pdf_path}")
        return cleaned_text
    
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        raise Exception(f"Failed to process PDF: {str(e)}")

def extract_metadata_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """Extract metadata from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "subject": metadata.get("subject", ""),
            "keywords": metadata.get("keywords", ""),
            "page_count": len(doc),
            "file_name": os.path.basename(pdf_path)
        }
    except Exception as e:
        logger.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
        return {"file_name": os.path.basename(pdf_path)}

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Define the end of this chunk
        end = start + chunk_size
        
        # Adjust to avoid cutting words in half
        if end < text_length:
            # Try to find a space or newline to break at
            while end > start and not text[end].isspace():
                end -= 1
            # If we couldn't find a good break, just use the original endpoint
            if end == start:
                end = start + chunk_size
        else:
            end = text_length
            
        # Extract this chunk
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
            
        # Move to next chunk with overlap
        start = end - overlap if end < text_length else text_length
    
    return chunks
