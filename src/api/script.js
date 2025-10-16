// src/api/script.js
let searchWebSocket = null;
let currentSearchId = null;

async function search() {
  const query = document.getElementById('searchInput').value.trim();
  if (!query) return;

  const depth = parseInt(document.getElementById('depth').value, 10) || 3;
  const maxResults = parseInt(document.getElementById('maxResults').value, 10) || 7;
  
  const resultsDiv = document.getElementById('results');
  const statusDiv = document.getElementById('status-log');
  const searchBtn = document.getElementById('searchBtn');

  currentSearchId = 'search-' + Date.now();

  searchBtn.disabled = true;
  searchBtn.textContent = 'Searching...';
  resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Initializing deep search...</p></div>';
  statusDiv.innerHTML = '';

  // Try WebSocket first, fallback to REST if it fails
  try {
    await initializeAndSearch(query, depth, maxResults);
  } catch (error) {
    console.error("WebSocket failed, using REST API:", error);
    await fallbackRestSearch(query, depth, maxResults, searchBtn);
  }
}

async function initializeAndSearch(query, depth, maxResults) {
  return new Promise((resolve, reject) => {
    initializeWebSocket();
    
    // Wait for connection and send message
    const checkAndSend = () => {
      if (searchWebSocket && searchWebSocket.readyState === WebSocket.OPEN) {
        const message = {
          type: "search",
          query: query,
          depth: depth,
          max_results: maxResults,
          session_id: currentSearchId
        };
        searchWebSocket.send(JSON.stringify(message));
        resolve();
      } else if (searchWebSocket && searchWebSocket.readyState === WebSocket.CONNECTING) {
        setTimeout(checkAndSend, 100);
      } else {
        reject(new Error("WebSocket connection failed"));
      }
    };
    
    setTimeout(checkAndSend, 100);
  });
}

function initializeWebSocket() {
  // Close existing connection
  if (searchWebSocket) {
    searchWebSocket.close();
  }

  // Create new WebSocket connection
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/search`;
  
  console.log("🔗 Connecting to WebSocket:", wsUrl);
  searchWebSocket = new WebSocket(wsUrl);

  searchWebSocket.onopen = function(event) {
    console.log('✅ WebSocket connected');
    updateStatus('Connected to search engine...', 'connected');
  };

  searchWebSocket.onmessage = function(event) {
    try {
      console.log("📥 Received WebSocket message:", event.data);
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    } catch (e) {
      console.error('Failed to parse WebSocket message:', e);
    }
  };

  searchWebSocket.onclose = function(event) {
    console.log('❌ WebSocket disconnected, code:', event.code);
    if (event.code !== 1000) {
      updateStatus('Connection lost. Please try again.', 'error');
    }
  };

  searchWebSocket.onerror = function(error) {
    console.error('WebSocket error:', error);
    updateStatus('Connection error. Please refresh the page.', 'error');
  };
}

function handleWebSocketMessage(data) {
  console.log("🔄 Handling WebSocket message type:", data.type);
  
  const statusDiv = document.getElementById('status-log');
  const resultsDiv = document.getElementById('results');
  const searchBtn = document.getElementById('searchBtn');

  switch (data.type) {
    case 'status':
      updateStatus(data.message, 'status');
      break;
      
    case 'progress':
      updateProgress(data.message, data.current, data.total);
      break;
      
    case 'complete':
      // Search completed, display results
      console.log("✅ Search completed, displaying results");
      displaySearchResults(data.data);
      resetSearchButton(searchBtn);
      break;
      
    case 'error':
      console.error("❌ Search error:", data.message);
      showError(data.message);
      resetSearchButton(searchBtn);
      break;
      
    default:
      console.log('Unknown message type:', data);
  }
}

// Fallback to REST API if WebSocket fails
async function fallbackRestSearch(query, depth, maxResults, searchBtn) {
  console.log("🔄 Using REST API fallback");
  updateStatus('Using standard search...', 'status');
  
  try {
    const response = await fetch('/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ 
        query: query, 
        depth: depth,
        max_results: maxResults 
      })
    });

    if (!response.ok) throw new Error('Search failed');
    const data = await response.json();
    
    console.log("✅ REST search completed");
    displaySearchResults(data);
    
  } catch (err) {
    console.error("❌ REST search failed:", err);
    showError(err.message);
  } finally {
    resetSearchButton(searchBtn);
  }
}

function updateStatus(message, type = 'status') {
  const statusDiv = document.getElementById('status-log');
  const div = document.createElement('div');
  div.className = `status-message ${type}`;
  div.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
  statusDiv.appendChild(div);
  statusDiv.scrollTop = statusDiv.scrollHeight;
}

function updateProgress(message, current, total) {
  const statusDiv = document.getElementById('status-log');
  const progress = total > 0 ? ` (${current}/${total})` : '';
  updateStatus(message + progress, 'progress');
}

function displaySearchResults(data) {
  const resultsDiv = document.getElementById('results');
  
  if (!data) {
    resultsDiv.innerHTML = '<div class="error">No results received</div>';
    return;
  }

  console.log("📄 Displaying search results, answer length:", data.answer?.length);

  let html = '<div class="result-card">';
  
  // ✅ CRITICAL FIX: Render HTML directly, don't escape it!
  if (data.answer && data.answer.length > 0) {
    // The answer is already formatted HTML from the backend
    html += data.answer;
  } else {
    html += '<div class="answer"><p>No answer returned.</p></div>';
  }

  // Sources block
  if (Array.isArray(data.sources) && data.sources.length > 0) {
    html += '<div class="sources">';
    html += '<div style="font-weight:700;margin-bottom:8px">📚 Sources (' + (data.total_sources || data.sources.length) + '):</div>';
    data.sources.forEach(source => {
      html += '<div class="source">';
      html += '<div class="source-title">' + escapeHtml(source.title || '') + '</div>';
      html += '<a href="' + escapeAttr(source.url || '#') + '" target="_blank" rel="noreferrer" class="source-url">' + escapeHtml(source.url || '') + '</a>';
      if (source.snippet) html += '<div class="source-snippet">' + escapeHtml(source.snippet) + '</div>';
      html += '</div>';
    });
    html += '</div>';
  }

  html += '</div>';
  resultsDiv.innerHTML = html;

  // Trigger MathJax typesetting if available
  if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
    try { 
      window.MathJax.typesetPromise(); 
    } catch (e) { 
      console.warn('MathJax typeset error', e); 
    }
  }
}

function showError(message) {
  const resultsDiv = document.getElementById('results');
  const searchBtn = document.getElementById('searchBtn');
  
  resultsDiv.innerHTML = '<div class="error">Error: ' + escapeHtml(message || 'Unknown') + '</div>';
  resetSearchButton(searchBtn);
}

function resetSearchButton(button) {
  if (button) {
    button.disabled = false;
    button.textContent = 'Search';
  }
}

// Helper functions
function escapeHtml(str) {
  if (str === undefined || str === null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function escapeAttr(str) {
  if (str === undefined || str === null) return '#';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// Close WebSocket when page unloads
window.addEventListener('beforeunload', function() {
  if (searchWebSocket) {
    searchWebSocket.close();
  }
});

// Add Enter key support
document.getElementById('searchInput').addEventListener('keypress', function(event) {
  if (event.key === 'Enter') {
    search();
  }
});