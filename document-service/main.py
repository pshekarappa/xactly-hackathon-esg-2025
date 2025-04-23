from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uuid
from typing import List, Dict, Any, Optional
import json
import glob
import logging
from pydantic import BaseModel

# Import our modules
from pdf_processor import extract_text_from_pdf, extract_metadata_from_pdf, chunk_text
from embedding_store import EmbeddingStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="ESG Document Service")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
UPLOAD_DIR = "pdfs"
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize embedding store
embedding_store = EmbeddingStore()

# Global storage for processing status
processing_status = {}

# Pydantic models for request/response
class SetupRequest(BaseModel):
    pdf_directory: Optional[str] = "pdfs"

class SearchRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5

class ProcessingStatusResponse(BaseModel):
    job_id: str
    status: str
    processed_count: int
    total_count: int
    failed_files: List[str]

@app.get("/")
def read_root():
    return {"status": "Document Service is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

def process_pdfs_task(job_id: str, pdf_directory: str):
    """Background task to process all PDFs in a directory."""
    try:
        # Find all PDF files in the directory
        pdf_files = glob.glob(f"{pdf_directory}/*.pdf")
        
        # Initialize status
        processing_status[job_id] = {
            "status": "processing",
            "processed_count": 0,
            "total_count": len(pdf_files),
            "failed_files": []
        }
        
        # Process each PDF
        for pdf_path in pdf_files:
            try:
                # Extract text and metadata
                text = extract_text_from_pdf(pdf_path)
                metadata = extract_metadata_from_pdf(pdf_path)
                
                # Generate a document ID
                doc_id = str(uuid.uuid4())
                
                # Chunk the text
                chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
                
                # Store in embedding database
                embedding_store.add_document_chunks(doc_id, chunks, metadata)
                
                # Update status
                processing_status[job_id]["processed_count"] += 1
                
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")
                processing_status[job_id]["failed_files"].append(os.path.basename(pdf_path))
        
        # Update final status
        processing_status[job_id]["status"] = "completed"
        
    except Exception as e:
        logger.error(f"Error in background processing task: {str(e)}")
        processing_status[job_id]["status"] = "failed"

@app.post("/setup", response_model=dict)
async def setup_document_service(
    background_tasks: BackgroundTasks,
    request: SetupRequest
):
    """
    Set up the document service by processing all PDFs in the specified directory.
    Returns a job ID that can be used to check the processing status.
    """
    pdf_directory = request.pdf_directory
    
    # Check if directory exists
    if not os.path.exists(pdf_directory):
        raise HTTPException(status_code=404, detail=f"Directory {pdf_directory} not found")
    
    # Generate a job ID
    job_id = str(uuid.uuid4())
    
    # Start processing in the background
    background_tasks.add_task(process_pdfs_task, job_id, pdf_directory)
    
    return {
        "job_id": job_id,
        "message": "PDF processing started",
        "status_endpoint": f"/status/{job_id}"
    }

@app.get("/status/{job_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(job_id: str):
    """Get the status of a PDF processing job."""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    
    status_data = processing_status[job_id]
    return {
        "job_id": job_id,
        **status_data
    }

@app.post("/search", response_model=List[Dict[str, Any]])
async def search_documents(request: SearchRequest):
    """
    Search for documents relevant to the query.
    Returns a list of document chunks ordered by relevance.
    """
    query = request.query
    n_results = request.n_results
    
    # Search for relevant documents
    results = embedding_store.search_documents(query, n_results)
    
    return results

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a single PDF file."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Process the file
        text = extract_text_from_pdf(file_path)
        metadata = extract_metadata_from_pdf(file_path)
        doc_id = str(uuid.uuid4())
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        embedding_store.add_document_chunks(doc_id, chunks, metadata)
        
        return {
            "message": f"File {file.filename} uploaded and processed successfully",
            "doc_id": doc_id,
            "chunk_count": len(chunks)
        }
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Run with: uvicorn main:app --reload
