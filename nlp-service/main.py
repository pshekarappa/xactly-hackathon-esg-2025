from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv

# Import our NLP processor
from nlp_processor import NLPProcessor

# Load environment variables (if any)
load_dotenv()

# Set the Document Service URL - modify this if your Document Service runs on a different URL
DOCUMENT_SERVICE_URL = os.getenv("DOCUMENT_SERVICE_URL", "http://localhost:8000")

# Initialize FastAPI app
app = FastAPI(title="ESG NLP Service")

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize our NLP processor
nlp_processor = NLPProcessor(document_service_url=DOCUMENT_SERVICE_URL)

# Define request model
class QueryRequest(BaseModel):
    query: str

# Define response model
class QueryResponse(BaseModel):
    results: list
    synthesized_answer: str = None

@app.get("/")
async def root():
    return {"message": "ESG NLP Service is running"}

@app.post("/process_query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Process the query using our NLP processor
        results, synthesized_answer = await nlp_processor.process_query(request.query)
        
        # Return the results
        return QueryResponse(
            results=results,
            synthesized_answer=synthesized_answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# For running directly with Python
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True) 