async function search() {
    const query = document.getElementById('searchInput').value;
    if (!query) return;
    
    const depth = parseInt(document.getElementById('depth').value);
    const maxResults = parseInt(document.getElementById('maxResults').value);
    const resultsDiv = document.getElementById('results');
    const searchBtn = document.getElementById('searchBtn');
    
    searchBtn.disabled = true;
    searchBtn.textContent = 'Going...';
    resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Deep searching across the web...</p></div>';
    
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
        
        let html = '<div class="result-card">';
        html += '<div class="answer">';
        html += '<h2>Answer:</h2>';
        html += '<p>' + data.answer.replace(/\n/g, '<br>') + '</p>';
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
