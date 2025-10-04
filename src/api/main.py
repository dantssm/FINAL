# src/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import asyncio
from datetime import datetime

from src.pipeline.deep_search import DeepSearchPipeline

# Initialize FastAPI app
app = FastAPI(
    title="AI Deep Search Engine with RAG",
    description="Deep search with AI-powered analysis and session-based knowledge",
    version="2.0.0"
)

# Enable CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the pipeline with session-based RAG
pipeline = DeepSearchPipeline()

# Store chat sessions (in production, use Redis)
chat_sessions = {}

# ===========================
# Request/Response Models
# ===========================

class SearchRequest(BaseModel):
    query: str
    depth: int = 3  # Deeper by default
    max_results: int = 7  # More results by default

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

# ===========================
# API Endpoints
# ===========================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Deep Search</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .search-box {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
            }
            input {
                flex: 1;
                padding: 15px;
                font-size: 16px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                outline: none;
                transition: all 0.3s;
            }
            input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            button {
                padding: 15px 30px;
                font-size: 16px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                transition: transform 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
            }
            button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            .options {
                display: flex;
                gap: 20px;
                margin-bottom: 20px;
                padding: 15px;
                background: #f5f5f5;
                border-radius: 10px;
            }
            .option {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            select {
                padding: 5px 10px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
            #results {
                margin-top: 20px;
            }
            .result-card {
                background: #f9f9f9;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
            }
            .answer {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                border-left: 4px solid #667eea;
            }
            .sources {
                margin-top: 20px;
            }
            .source {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                border: 1px solid #e0e0e0;
                transition: all 0.3s;
            }
            .source:hover {
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            .source-title {
                font-weight: bold;
                color: #333;
                margin-bottom: 5px;
            }
            .source-url {
                color: #667eea;
                text-decoration: none;
                font-size: 14px;
            }
            .source-snippet {
                color: #666;
                font-size: 14px;
                margin-top: 5px;
            }
            .loading {
                text-align: center;
                padding: 40px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .error {
                background: #fee;
                color: #c00;
                padding: 15px;
                border-radius: 10px;
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 AI Deep Search Engine</h1>
            
            <div class="search-box">
                <input 
                    type="text" 
                    id="searchInput" 
                    placeholder="Ask anything... (e.g., 'How does quantum computing work?')"
                    onkeypress="if(event.key === 'Enter') search()"
                >
                <button onclick="search()" id="searchBtn">Search</button>
            </div>
            
            <div class="options">
                <div class="option">
                    <label>Search Depth:</label>
                    <select id="depth">
                        <option value="1">Quick (1 level)</option>
                        <option value="2">Standard (2 levels)</option>
                        <option value="3" selected>Deep (3 levels)</option>
                        <option value="4">Very Deep (4 levels)</option>
                        <option value="5">Maximum (5 levels)</option>
                    </select>
                </div>
                <div class="option">
                    <label>Max Results:</label>
                    <select id="maxResults">
                        <option value="3">3</option>
                        <option value="5">5</option>
                        <option value="7" selected>7</option>
                        <option value="10">10</option>
                        <option value="15">15</option>
                    </select>
                </div>
            </div>
            
            <div id="results"></div>
        </div>
        
        <script>
            async function search() {
                const query = document.getElementById('searchInput').value;
                if (!query) return;
                
                const depth = parseInt(document.getElementById('depth').value);
                const maxResults = parseInt(document.getElementById('maxResults').value);
                const resultsDiv = document.getElementById('results');
                const searchBtn = document.getElementById('searchBtn');
                
                // Show loading
                searchBtn.disabled = true;
                searchBtn.textContent = 'Searching...';
                resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Deep searching across the web...</p></div>';
                
                try {
                    const response = await fetch('/search', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            query: query,
                            depth: depth,
                            max_results: maxResults
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Search failed');
                    }
                    
                    const data = await response.json();
                    
                    // Display results
                    let html = '<div class="result-card">';
                    html += '<div class="answer">';
                    html += '<h2>Answer:</h2>';
                    html += '<p>' + data.answer.replace(/\\n/g, '<br>') + '</p>';
                    html += '</div>';
                    
                    if (data.sources && data.sources.length > 0) {
                        html += '<div class="sources">';
                        html += '<h3>📚 Sources (' + data.total_sources + ' found):</h3>';
                        
                        data.sources.forEach(source => {
                            html += '<div class="source">';
                            html += '<div class="source-title">' + source.title + '</div>';
                            html += '<a href="' + source.url + '" target="_blank" class="source-url">' + source.url + '</a>';
                            if (source.snippet) {
                                html += '<div class="source-snippet">' + source.snippet + '</div>';
                            }
                            html += '</div>';
                        });
                        
                        html += '</div>';
                    }
                    
                    html += '</div>';
                    resultsDiv.innerHTML = html;
                    
                } catch (error) {
                    resultsDiv.innerHTML = '<div class="error">Error: ' + error.message + '</div>';
                } finally {
                    searchBtn.disabled = false;
                    searchBtn.textContent = 'Search';
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform deep search
    """
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
    """
    Chat with context
    """
    try:
        # Get or create session
        if request.session_id not in chat_sessions:
            chat_sessions[request.session_id] = []
        
        # Get response
        response = await pipeline.chat(
            message=request.message,
            use_search=request.use_search
        )
        
        # Store in session
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
    """Check if API is running"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/knowledge-stats")
async def knowledge_stats():
    """Get knowledge base statistics"""
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