# src/api/main.py - FastAPI server
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import os
import uuid

from src.pipeline.deep_search import DeepSearchPipeline
from src.api.websocket_manager import manager

app = FastAPI(
    title="AI Deep Search Engine - OPTIMIZED FREE",
    description="Lightning-fast deep search with FREE services only",
    version="3.0.1"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

# Initialize pipeline
pipeline = DeepSearchPipeline()
chat_sessions = {}

# Models
class SearchRequest(BaseModel):
    query: str
    depth: int = 3
    max_results: int = 7

class ChatRequest(BaseModel):
    message: str
    session_id: str
    use_search: bool = True

class SearchResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict]
    total_sources: int
    search_depth: int
    timestamp: str
    cache_stats: Optional[Dict] = None

class ChatResponse(BaseModel):
    message: str
    response: str
    session_id: str
    timestamp: str

# WebSocket Routes
@app.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            print(f"📥 Received: {data}")
            
            if data.get("type") == "search":
                query = data.get("query", "")
                depth = data.get("depth", 3)
                max_results = data.get("max_results", 7)
                session_id = data.get("session_id", "")
                
                if not query:
                    await manager.send_message(websocket, {
                        "type": "error",
                        "message": "Query is required"
                    })
                    continue
                
                print(f"🔍 WebSocket search: {query}")
                print(f"   Depth: {depth} (will generate {depth + 1} queries)")
                print(f"   Max results: {max_results} per query")
                
                try:
                    # Use HTML format for WebSocket (web interface)
                    result = await pipeline.search(
                        query=query,
                        session_id=session_id,
                        depth=depth,
                        max_results_per_search=max_results,
                        websocket=websocket,
                        return_format="html"  # HTML for web display
                    )
                    
                    # Result already sent via WebSocket
                    
                except Exception as e:
                    await manager.send_message(websocket, {
                        "type": "error",
                        "message": f"Search failed: {str(e)}"
                    })
            
            # Handle session end
            elif data.get("type") == "end_session":
                session_id = data.get("session_id", "")
                if session_id:
                    print(f"👋 Ending session: {session_id[:12]}...")
                    pipeline.end_session(session_id)
                    await manager.send_message(websocket, {
                        "type": "session_ended",
                        "message": "Session ended and cache cleared"
                    })
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# REST Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve HTML page"""
    html_path = os.path.join(BASE_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform deep search (REST endpoint)"""
    try:
        print(f"\n🔍 REST API search: {request.query}")
        print(f"   Depth: {request.depth} (will generate {request.depth + 1} queries)")
        print(f"   Max results: {request.max_results} per query")
        
        # Use text format for REST API (no HTML tags)
        result = await pipeline.search(
            query=request.query,
            depth=request.depth,
            max_results_per_search=request.max_results,
            return_format="text"  # Plain text for API
        )
        return SearchResponse(
            query=result['query'],
            answer=result['answer'],
            sources=result['sources'],
            total_sources=result['total_sources'],
            search_depth=result['search_depth'],
            timestamp=datetime.now().isoformat(),
            cache_stats=result.get('cache_stats')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with context"""
    try:
        if request.session_id not in chat_sessions:
            chat_sessions[request.session_id] = []
        
        # Use text format for chat API
        response = await pipeline.chat(
            message=request.message,
            session_id=request.session_id,
            use_search=request.use_search,
            return_format="text"  # Plain text for API
        )
        
        chat_sessions[request.session_id].append({
            "message": request.message,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return ChatResponse(
            message=request.message,
            response=response,
            session_id=request.session_id,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/session/{session_id}")
async def end_session(session_id: str):
    """End a session and clear its cache"""
    try:
        print(f"\n👋 API request to end session: {session_id[:12]}...")
        pipeline.end_session(session_id)
        
        if session_id in chat_sessions:
            del chat_sessions[session_id]
        
        return {
            "status": "success",
            "message": f"Session {session_id[:12]} ended and cache cleared",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check with cache stats"""
    stats = pipeline.get_knowledge_stats()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_stats": stats.get('cache_stats'),
        "active_sessions": stats.get('active_sessions')
    }

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    stats = pipeline.get_knowledge_stats()
    return {
        "cache": stats.get('cache_stats'),
        "active_sessions": stats.get('active_sessions'),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/cache/clear")
async def clear_cache():
    """Clear cache (admin endpoint)"""
    await pipeline.cache.clear_all()
    return {
        "status": "success",
        "message": "All cache cleared",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/knowledge-stats")
async def knowledge_stats():
    """Get knowledge base statistics"""
    stats = pipeline.get_knowledge_stats()
    return {
        "cache_stats": stats.get('cache_stats'),
        "active_sessions": stats.get('active_sessions'),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting OPTIMIZED AI Deep Search API (FREE)...")
    print("📍 Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)