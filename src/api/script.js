// /static/script.js
async function search() {
  const query = document.getElementById('searchInput').value.trim();
  if (!query) return;

  const depth = parseInt(document.getElementById('depth').value, 10) || 3;
  const maxResults = parseInt(document.getElementById('maxResults').value, 10) || 7;
  const resultsDiv = document.getElementById('results');
  const searchBtn = document.getElementById('searchBtn');

  searchBtn.disabled = true;
  searchBtn.textContent = 'Going...';
  resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Deep searching across the web...</p></div>';

  try {
    const response = await fetch('/search', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ query, depth, max_results: maxResults })
    });

    if (!response.ok) throw new Error('Search failed');

    const data = await response.json();
    const raw = data.answer || '';

    // 1) Normalize: strip Markdown markers but keep LaTeX math intact
    const normalized = stripMarkdownKeepMath(raw);

    // 2) Split into paragraphs, trim, dedupe identical paragraphs while preserving order
    const paragraphs = normalized
      .split(/\n\s*\n/)
      .map(p => p.trim())
      .filter(p => p.length > 0);

    const seen = new Set();
    const uniqueParagraphs = [];
    for (const p of paragraphs) {
      if (!seen.has(p)) {
        seen.add(p);
        uniqueParagraphs.push(p);
      }
    }

    // 3) Build HTML output
    let html = '<div class="result-card">';
    html += '<div class="answer">';

    if (uniqueParagraphs.length === 0) {
      html += '<p>No answer returned.</p>';
    } else {
      uniqueParagraphs.forEach(p => {
        const safe = escapeHtmlExceptMath(p).replace(/\n/g, '<br>');
        html += '<p>' + safe + '</p>';
      });
    }

    html += '</div>';

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

    // 4) Trigger MathJax typesetting if present
    if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
      try { await window.MathJax.typesetPromise(); } catch (e) { console.warn('MathJax typeset error', e); }
    }
  } catch (err) {
    resultsDiv.innerHTML = '<div class="error">Error: ' + escapeHtml(err.message || 'Unknown') + '</div>';
  } finally {
    searchBtn.disabled = false;
    searchBtn.textContent = 'Search';
  }
}

// ---- Helpers ----

// Strip Markdown headings and bold markers, preserve math ($...$ and $$...$$)
function stripMarkdownKeepMath(text) {
  if (!text) return '';

  // extract math regions
  const mathRegions = [];
  const mathRe = /(\$\$[\s\S]+?\$\$|\$[^$\n]+\$)/g;
  let idx = 0;
  const placeholderText = text.replace(mathRe, m => {
    const key = `@@MATH${idx}@@`;
    mathRegions.push({ key, math: m });
    idx++;
    return key;
  });

  // remove leading Markdown headers and bold/italic markers
  const cleanedLines = placeholderText.split('\n').map(line => {
    // remove leading hashes and a following space
    line = line.replace(/^\s{0,3}#{1,6}\s*/, '');
    // remove bold/italic markers but keep text
    line = line.replace(/\*\*(.*?)\*\*/g, '$1');
    line = line.replace(/__(.*?)__/g, '$1');
    line = line.replace(/\*(.*?)\*/g, '$1');
    line = line.replace(/_(.*?)_/g, '$1');
    // remove backticks for inline code
    line = line.replace(/`([^`]+)`/g, '$1');
    return line;
  }).join('\n');

  // restore math regions
  let restored = cleanedLines;
  for (const r of mathRegions) {
    restored = restored.replace(r.key, r.math);
  }
  return restored;
}

// Escape HTML but keep math regions intact
function escapeHtmlExceptMath(text) {
  if (!text) return '';
  const mathRegions = [];
  const mathRe = /(\$\$[\s\S]+?\$\$|\$[^$\n]+\$)/g;
  let idx = 0;
  const withPlaceholders = text.replace(mathRe, m => {
    const key = `@@M${idx}@@`;
    mathRegions.push({ key, math: m });
    idx++;
    return key;
  });

  let escaped = withPlaceholders
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');

  for (const r of mathRegions) {
    escaped = escaped.replace(r.key, r.math);
  }
  return escaped;
}

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
