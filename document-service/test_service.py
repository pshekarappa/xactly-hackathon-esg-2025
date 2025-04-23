import requests
import json
import time
import os
import sys

BASE_URL = "http://localhost:8000"

def test_setup():
    print("Testing /setup endpoint...")
    
    # Make sure we have PDFs in the directory
    if not any(f.endswith('.pdf') for f in os.listdir('pdfs')):
        print("No PDF files found in 'pdfs' directory. Please add some PDFs before testing.")
        sys.exit(1)
    
    # Call setup endpoint
    response = requests.post(
        f"{BASE_URL}/setup",
        json={"pdf_directory": "pdfs"}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Setup started successfully. Job ID: {result['job_id']}")
        return result['job_id']
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def check_status(job_id):
    print(f"Checking status for job {job_id}...")
    
    # Poll status until completed
    while True:
        response = requests.get(f"{BASE_URL}/status/{job_id}")
        
        if response.status_code == 200:
            status = response.json()
            print(f"Status: {status['status']} - Processed: {status['processed_count']}/{status['total_count']}")
            
            if status['status'] in ['completed', 'failed']:
                if status['failed_files']:
                    print(f"Failed files: {', '.join(status['failed_files'])}")
                break
                
        else:
            print(f"Error checking status: {response.status_code} - {response.text}")
            break
            
        time.sleep(2)  # Wait 2 seconds before checking again

def test_search():
    print("\nTesting /search endpoint...")
    
    # Example queries
    queries = [
        "carbon emissions policy",
        "sustainable supply chain",
        "diversity and inclusion initiatives"
    ]
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        
        response = requests.post(
            f"{BASE_URL}/search",
            json={"query": query, "n_results": 3}
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"Found {len(results)} results")
            
            for i, result in enumerate(results):
                print(f"\nResult {i+1}:")
                print(f"Score: {result['score']}")
                print(f"Source: {result['metadata'].get('file_name', 'Unknown')}")
                print(f"Text snippet: {result['text'][:150]}...")
        else:
            print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    job_id = test_setup()
    if job_id:
        check_status(job_id)
        test_search()
