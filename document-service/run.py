import uvicorn
import argparse

def run_server(host="0.0.0.0", port=8000, reload=True):
    """
    Run the FastAPI application using uvicorn.
    
    Args:
        host: Host address to bind to
        port: Port to bind to
        reload: Whether to enable auto-reload on code changes
    """
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    # Set up argument parser for command line options
    parser = argparse.ArgumentParser(description="Run the ESG Document Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload on code changes")
    
    args = parser.parse_args()
    
    print(f"Starting ESG Document Service on {args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    # Run the server with the provided arguments
    run_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload
    ) 