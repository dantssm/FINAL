"""
FastAPI server for deep search
Simplified to match the student version pipeline
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
import os

from src.pipeline.deep_search import DeepSearchPipeline

app = FastAPI(
    title="AI Deep Search Engine",
    description="Deep search using Google, web scraping, and AI",
    version="1.0"
)

# Enable CORS so frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS, images, etc.)
# The files are directly in src/api/ not in a static subfolder
BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

# Initialize the search pipeline once when server starts
print("🚀 Starting Deep Search API...")
pipeline = DeepSearchPipeline()
print("✅ API ready!\n")


# Request/Response models (defines what the API expects and returns)
class SearchRequest(BaseModel):
    query: str
    depth: int = 2  # How many search queries to generate
    max_results: int = 7  # Results per query

class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict]
    total_sources: int
    chunks_analyzed: int
    time_seconds: float
    timestamp: str


# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    """
    Serve the main web interface HTML page
    This is what shows up when you visit http://localhost:8000/
    """
    html_path = os.path.join(BASE_DIR, "index.html")
    
    # Check if the HTML file exists
    if not os.path.exists(html_path):
        return HTMLResponse(
            content="""
            <html>
                <head><title>Deep Search Engine</title></head>
                <body>
                    <h1>Deep Search Engine API</h1>
                    <p>The API is running, but the web interface (index.html) was not found.</p>
                    <p>Available endpoints:</p>
                    <ul>
                        <li>WebSocket: ws://localhost:8000/ws/search</li>
                        <li>REST API: POST /api/search</li>
                        <li>Health Check: GET /api/health</li>
                        <li>API Docs: <a href="/docs">/docs</a></li>
                    </ul>
                </body>
            </html>
            """,
            status_code=200
        )
    
    # Serve the HTML file
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# WebSocket endpoint - for real-time updates in the browser
@app.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    """
    WebSocket connection for search with real-time progress updates
    
    This is what lets the frontend show "Searching Google..." messages
    while the search is running instead of just showing a loading spinner
    """
    await websocket.accept()
    print("🔌 WebSocket connected")
    
    try:
        # Keep listening for messages from the frontend
        while True:
            data = await websocket.receive_json()
            print(f"📥 Received WebSocket message: {data.get('type')}")
            
            if data.get("type") == "search":
                query = data.get("query", "")
                depth = data.get("depth", 2)
                max_results = data.get("max_results", 7)
                
                if not query:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Query is required"
                    })
                    continue
                
                print(f"\n🔍 WebSocket search started")
                print(f"   Query: '{query}'")
                print(f"   Depth: {depth}")
                print(f"   Max results: {max_results}")
                
                try:
                    # Run the search - the websocket parameter lets it send updates
                    result = await pipeline.search(
                        query=query,
                        depth=depth,
                        max_results=max_results,
                        websocket=websocket
                    )
                    
                    # Send the final result with type "complete" so frontend recognizes it
                    await websocket.send_json({
                        "type": "complete",
                        "data": {
                            "query": result["query"],
                            "answer": result["answer"],
                            "sources": result["sources"],
                            "total_sources": result["total_sources"],
                            "chunks_analyzed": result["chunks_analyzed"],
                            "time_seconds": result["time_seconds"]
                        }
                    })
                    
                    print(f"✅ WebSocket search complete\n")
                    
                except Exception as e:
                    print(f"❌ Search error: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Search failed: {str(e)}"
                    })
    
    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {str(e)}")


# REST API endpoint - for programmatic access without WebSocket
@app.post("/api/search", response_model=SearchResponse)
async def search_api(request: SearchRequest):
    """
    REST API endpoint for search
    
    This is for when you want to use the search engine from code/scripts
    instead of through the web interface. No real-time updates, just
    makes the request and waits for the final result.
    """
    try:
        print(f"\n🔍 REST API search started")
        print(f"   Query: '{request.query}'")
        print(f"   Depth: {request.depth}")
        print(f"   Max results: {request.max_results}")
        
        # Run search without websocket (no progress updates)
        result = await pipeline.search(
            query=request.query,
            depth=request.depth,
            max_results=request.max_results,
            websocket=None  # No websocket = no progress updates
        )
        
        print(f"✅ REST API search complete\n")
        
        return SearchResponse(
            query=result['query'],
            answer=result['answer'],
            sources=result['sources'],
            total_sources=result['total_sources'],
            chunks_analyzed=result['chunks_analyzed'],
            time_seconds=result['time_seconds'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Check if the API is running"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "Deep Search API is running"
    }


@app.post("/api/clear")
async def clear_knowledge():
    """Clear the vector store (start fresh)"""
    try:
        pipeline.clear_knowledge()
        return {
            "status": "success",
            "message": "Knowledge base cleared",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the server
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("🚀 Starting Deep Search API Server")
    print("="*60)
    print("📍 API will be available at: http://localhost:8000")
    print("📍 WebSocket endpoint: ws://localhost:8000/ws/search")
    print("📍 REST API endpoint: http://localhost:8000/api/search")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)