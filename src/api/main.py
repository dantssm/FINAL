# src/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime
import os
import asyncio

from src.pipeline.deep_search import DeepSearchPipeline

app = FastAPI(
    title="AI Deep Search Engine with RAG",
    description="Deep search with AI-powered analysis and session-based knowledge",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS)
BASE_DIR = os.path.dirname(__file__)
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

pipeline = DeepSearchPipeline()
chat_sessions = {}

# -------------------------------
# Models
# -------------------------------
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

class ChatResponse(BaseModel):
    message: str
    response: str
    session_id: str
    timestamp: str

# -------------------------------
# Routes
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve HTML page"""
    html_path = os.path.join(BASE_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform deep search"""
    try:
        result = await pipeline.search(
            query=request.query,
            depth=request.depth,
            max_results_per_search=request.max_results
        )
        return SearchResponse(
            query=result['query'],
            answer=result['answer'],
            sources=result['sources'],
            total_sources=result['total_sources'],
            search_depth=result['search_depth'],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with context"""
    try:
        if request.session_id not in chat_sessions:
            chat_sessions[request.session_id] = []
        response = await pipeline.chat(
            message=request.message,
            use_search=request.use_search
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

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/knowledge-stats")
async def knowledge_stats():
    stats = pipeline.get_knowledge_stats()
    return {
        "total_documents": stats['total_documents'],
        "search_history_length": stats['search_history_length'],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Deep Search API...")
    print("📍 Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
