from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import requests
import json

# Create a FastAPI app
app = FastAPI(title="NLP Service")

# Load the sentence transformer model
# Note: This will download the model on first run (about 500MB)
model = SentenceTransformer('all-MiniLM-L6-v2')  # A smaller, faster model good for beginners

# Define the data model for incoming queries
class Query(BaseModel):
    query: str

# Define the data model for outgoing responses
class Response(BaseModel):
    answer: str
    source_chunks: list

# Document service URL (adjust if needed)
DOCUMENT_SERVICE_URL = "http://localhost:8001"  # Assuming Document service runs on port 8001

@app.get("/")
def read_root():
    return {"message": "NLP Service is running"}

@app.post("/process_query")
async def process_query(query_data: Query):
    try:
        # Get the user's query
        user_query = query_data.query
        
        # Generate embedding for the query using the sentence transformer
        query_embedding = model.encode(user_query).tolist()
        
        # Prepare payload for the document service
        payload = {
            "query_embedding": query_embedding,
            "top_k": 3  # Return top 3 most relevant chunks
        }
        
        # Make HTTP request to the Document Service
        response = requests.post(
            f"{DOCUMENT_SERVICE_URL}/search",
            json=payload
        )
        
        # Check if the request was successful
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Error communicating with Document Service")
        
        # Parse the response from the Document Service
        document_results = response.json()
        text_chunks = document_results.get("chunks", [])
        
        # Simple approach: Just concatenate the chunks with some formatting
        if text_chunks:
            # Join the chunks with newlines in between
            answer = "\n\n".join([f"• {chunk}" for chunk in text_chunks])
            
            # Create the response
            result = {
                "answer": answer,
                "source_chunks": text_chunks
            }
            
            return result
        else:
            return {
                "answer": "I couldn't find any relevant information about your query.",
                "source_chunks": []
            }
            
    except Exception as e:
        # Handle any errors
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Run the app with: uvicorn main:app --reload --port 8002
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
