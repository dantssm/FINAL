import uvicorn

if __name__ == "__main__":
    print("--- AI DEEP SEARCH ENGINE ---")
    print("\nOpen your browser and go to:")
    print("http://localhost:8000")
    print("\nAPI Documentation available at:")
    print("http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    
    uvicorn.run("src.api.main:app",
                host="0.0.0.0",
                port=8000,
                reload=True)