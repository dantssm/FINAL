# src/pipeline/deep_search.py - FIXED: Proper depth and max_results handling
"""
Deep Search Pipeline with proper text/HTML formatting and fixed parameters
"""

import asyncio
from typing import List, Dict, Optional
import hashlib
import os
import shutil
from datetime import datetime, timedelta
from fastapi import WebSocket
import re
from urllib.parse import urlparse

from src.config import config
from src.cache.memory_cache import MemoryCache, CachedGoogleSearcher, CachedJinaScraper
from src.llm.openrouter_client import OpenRouterClient
from src.rag.vector_store import VectorStore


def format_answer_html(raw_answer: str, sources: list) -> str:
    """
    Format answer as HTML for web display with CLEAR source citations.
    Shows source numbers with domains and full reference list at the end.
    """
    if not raw_answer:
        return ""

    # Build source map: Source N → {url, title, domain}
    source_map = {}
    for i, r in enumerate(sources, 1):
        url = r.get("url") or r.get("link") or "#"
        title = r.get("title", "Untitled")
        
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        
        source_map[i] = {
            "url": url,
            "title": title,
            "domain": domain
        }

    # Extract inline URLs from "Source N(https://...)" and update map
    def extract_inline_urls(match):
        n = int(match.group(1))
        url = match.group(2).strip()
        if n in source_map:
            source_map[n]["url"] = url
        return f"Source {n}"

    cleaned_text = re.sub(
        r"Source\s+(\d+)\s*\(\s*(https?://[^\s)]+)\s*\)",
        extract_inline_urls,
        raw_answer
    )

    # Remove any remaining bare URLs
    cleaned_text = re.sub(r'https?://[^\s)]+', '', cleaned_text)

    # Remove artifact tags
    cleaned_text = re.sub(r'<｜begin▁of▁sentence｜>', '', cleaned_text)
    cleaned_text = re.sub(r'<｜end▁of▁sentence｜>', '', cleaned_text)

    # Split into paragraphs
    paragraphs = [p.strip() for p in cleaned_text.split("\n\n") if p.strip()]

    html = "<div class='chat-block'>"

    # Track which sources were actually referenced
    referenced_sources = set()

    for p in paragraphs:
        # Clean markdown formatting
        clean_p = re.sub(r"[*_#>`]", "", p).strip()
        
        html += f"<p>{clean_p}</p>"

        # Find all Source N mentions
        found = re.findall(r"Source\s+(\d+)", clean_p)
        found_unique = sorted(set(int(n) for n in found if n.isdigit()))
        
        # Track referenced sources
        referenced_sources.update(found_unique)

        # Add inline source buttons after paragraph (with numbers!)
        if found_unique:
            html += "<div class='chat-sources' style='margin: 8px 0;'>"
            for n in found_unique:
                if n not in source_map:
                    continue
                    
                source = source_map[n]
                url = source["url"]
                domain = source["domain"]
                
                if not url.startswith("http"):
                    continue

                # Button with source number AND domain
                html += (
                    f'<a href="{url}" class="chat-source-btn" '
                    f'title="{source["title"]}" '
                    f'target="_blank" rel="noopener noreferrer">'
                    f'[{n}] {domain}</a>'
                )
            html += "</div>"

    # Add comprehensive "Sources Referenced" section at the end
    if referenced_sources:
        html += "<div style='margin-top: 24px; padding-top: 16px; border-top: 2px solid #e5e7eb;'>"
        html += "<h3 style='font-size: 16px; font-weight: 700; margin-bottom: 12px; color: #374151;'>📚 Sources Referenced:</h3>"
        
        for n in sorted(referenced_sources):
            if n not in source_map:
                continue
                
            source = source_map[n]
            url = source["url"]
            title = source["title"]
            domain = source["domain"]
            
            if not url.startswith("http"):
                continue
            
            # Each source with number, title, and domain
            html += f"<div style='margin-bottom: 10px; padding: 8px; background: #f9fafb; border-radius: 6px;'>"
            html += f"<div style='font-weight: 600; color: #8b5cf6;'>[{n}] {title[:80]}</div>"
            html += (
                f"<a href='{url}' target='_blank' rel='noopener noreferrer' "
                f"style='font-size: 13px; color: #6b7280; text-decoration: none;'>"
                f"🔗 {domain}</a>"
            )
            html += "</div>"
        
        html += "</div>"

    html += "</div>"
    return html


def format_answer_text(raw_answer: str) -> str:
    """
    Format answer as plain text (no HTML).
    Removes HTML tags and cleans up formatting.
    """
    if not raw_answer:
        return ""

    # Remove HTML tags if any
    text = re.sub(r'<[^>]+>', '', raw_answer)
    
    # Remove markdown formatting
    text = re.sub(r'[*_#>`]', '', text)
    
    # Remove artifact tags
    text = re.sub(r'<｜begin▁of▁sentence｜>', '', text)
    text = re.sub(r'<｜end▁of▁sentence｜>', '', text)
    
    # Clean up Source N(URL) references - keep just "Source N"
    text = re.sub(r'Source\s+(\d+)\s*\(https?://[^\s)]+\)', r'Source \1', text)
    
    # Remove any remaining bare URLs
    text = re.sub(r'https?://[^\s]+', '', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


class SearchLogger:
    """Tracks search progress with detailed logging"""
    
    def __init__(self):
        self.steps = []
        self.start_time = datetime.now()
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        elapsed = (datetime.now() - self.start_time).total_seconds()
        formatted = f"[{timestamp}] [{elapsed:.2f}s] {message}"
        self.steps.append(formatted)
        
        # Color coding for console
        if level == "SUCCESS":
            print(f"✅ {formatted}")
        elif level == "WARNING":
            print(f"⚠️  {formatted}")
        elif level == "ERROR":
            print(f"❌ {formatted}")
        elif level == "PROGRESS":
            print(f"🔄 {formatted}")
        else:
            print(f"📝 {formatted}")
    
    def get_steps(self):
        return self.steps


class SessionManager:
    """Manages session-based vector stores"""
    
    def __init__(self, base_path: str = "./data/sessions"):
        self.base_path = base_path
        self.sessions = {}
        self.session_last_access = {}
        
        os.makedirs(base_path, exist_ok=True)
        self._cleanup_old_sessions()
        print(f"📁 Session Manager initialized at: {base_path}")
    
    def get_or_create_session(self, session_id: str) -> VectorStore:
        if session_id not in self.sessions:
            session_path = os.path.join(self.base_path, session_id)
            os.makedirs(session_path, exist_ok=True)
            self.sessions[session_id] = VectorStore(persist_directory=session_path)
            print(f"📁 Created new session: {session_id[:12]}... at {session_path}")
        
        self.session_last_access[session_id] = datetime.now()
        return self.sessions[session_id]
    
    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            
            session_path = os.path.join(self.base_path, session_id)
            if os.path.exists(session_path):
                shutil.rmtree(session_path)
                print(f"🗑️ Deleted session: {session_id[:12]}...")
        
        if session_id in self.session_last_access:
            del self.session_last_access[session_id]
    
    def _cleanup_old_sessions(self, max_age_hours: int = 24):
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        if os.path.exists(self.base_path):
            for session_dir in os.listdir(self.base_path):
                session_path = os.path.join(self.base_path, session_dir)
                
                if os.path.isdir(session_path):
                    mtime = datetime.fromtimestamp(os.path.getmtime(session_path))
                    if mtime < cutoff_time:
                        shutil.rmtree(session_path)
                        cleaned_count += 1
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned {cleaned_count} old session(s) (>{max_age_hours}h)")
    
    def get_active_sessions_count(self) -> int:
        return len(self.sessions)


class DeepSearchPipeline:
    """
    Deep Search Pipeline with proper text/HTML formatting
    FIXED: Depth and max_results now work correctly!
    """
    
    def __init__(self):
        print("\n" + "="*60)
        print("🚀 INITIALIZING OPTIMIZED DEEP SEARCH PIPELINE")
        print("="*60)
        
        config.validate()
        
        # Initialize cache WITHOUT persistence
        print("\n📦 Setting up session-based cache system...")
        self.cache = MemoryCache(
            default_ttl_hours=config.CACHE_TTL_HOURS,
            max_size=config.MAX_CACHE_SIZE
        )
        print(f"   ✅ Session cache ready: {config.MAX_CACHE_SIZE} entries max")
        
        # Initialize cached searcher
        print("\n🔍 Setting up Google Search...")
        self.google_searcher = CachedGoogleSearcher(
            config.GOOGLE_SEARCH_API_KEY,
            config.GOOGLE_CSE_ID,
            self.cache
        )
        print(f"   ✅ Google Search ready with session caching")
        
        # Use Jina AI Reader with connection pooling
        print("\n🌐 Setting up web scraper...")
        self.web_scraper = CachedJinaScraper(self.cache)
        print(f"   ✅ Jina AI Reader ready with connection pooling")
        
        # Use FREE OpenRouter model
        print("\n🤖 Setting up LLM client...")
        self.llm_client = OpenRouterClient(
            config.OPENROUTER_API_KEY,
            model_name=config.OPENROUTER_MODEL
        )
        model_info = self.llm_client.get_model_info()
        print(f"   ✅ LLM ready: {model_info['model']}")
        print(f"   📋 Fallback models: {', '.join(model_info['fallback_models'][:2])}...")
        
        # Session manager for RAG stores
        print("\n💾 Setting up session management...")
        self.session_manager = SessionManager()
        self.session_histories = {}
        print(f"   ✅ Session manager ready")
        
        print("\n" + "="*60)
        print("✅ OPTIMIZED DEEP SEARCH PIPELINE READY!")
        print("="*60 + "\n")
    
    async def _send_websocket_progress(self, websocket: Optional[WebSocket], message: dict):
        """Send WebSocket message without blocking"""
        if not websocket:
            return
        
        try:
            await websocket.send_json(message)
        except Exception as e:
            # Silent fail - don't break search process
            pass
    
    def generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = str(datetime.now().timestamp())
        random_str = os.urandom(8).hex()
        session_id = hashlib.md5(f"{timestamp}{random_str}".encode()).hexdigest()[:16]
        print(f"🆔 Generated new session ID: {session_id}")
        return session_id
    
    async def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        depth: int = 2,
        max_results_per_search: int = 7,
        use_rag: bool = True,
        websocket: Optional[WebSocket] = None,
        return_format: str = "html"  # "html" or "text"
    ) -> Dict:
        """
        Deep search with proper formatting
        FIXED: depth and max_results_per_search now work correctly!
        
        Args:
            depth: Number of search levels (1-5). Controls how many diverse queries are generated.
            max_results_per_search: Maximum results to fetch per search query (1-15)
            return_format: "html" for web display, "text" for plain text API responses
        """
        logger = SearchLogger()
        
        await self._send_websocket_progress(websocket, {
            "type": "status",
            "message": "🚀 Starting optimized search...",
            "stage": "starting"
        })
        
        if not session_id:
            session_id = self.generate_session_id()
        
        logger.log("="*60)
        logger.log("🔍 STARTING OPTIMIZED DEEP SEARCH", "SUCCESS")
        logger.log("="*60)
        logger.log(f"Query: '{query}'")
        logger.log(f"Session ID: {session_id[:12]}...")
        logger.log(f"Search Depth: {depth} levels (will generate {depth + 1} diverse queries)")
        logger.log(f"Max Results per Search: {max_results_per_search}")
        logger.log(f"Using RAG: {use_rag}")
        logger.log(f"Return format: {return_format}")
        logger.log(f"Using Model: {config.OPENROUTER_MODEL}")
        logger.log(f"Using Embeddings: {config.EMBEDDING_PROVIDER}")
        logger.log("="*60)
        
        vector_store = self.session_manager.get_or_create_session(session_id)
        logger.log(f"📚 Vector store ready: {vector_store.get_stats()['total_documents']} existing documents")
        
        all_search_results = []
        
        # FIXED: Use user's depth parameter directly
        await self._send_websocket_progress(websocket, {
            "type": "status",
            "message": "🧠 Planning search strategy...",
            "stage": "planning"
        })
        
        logger.log("\n🧠 PHASE 1: Query Generation", "PROGRESS")
        num_queries = depth + 1  # depth=1 -> 2 queries, depth=3 -> 4 queries, etc.
        logger.log(f"Generating {num_queries} diverse search queries based on depth={depth}...")
        
        # Generate queries using improved LLM method
        search_queries = await self.llm_client.generate_search_queries(
            query,
            num_queries=num_queries
        )
        search_queries.insert(0, query)  # Original query first
        
        # Deduplicate immediately
        unique_queries = []
        seen = set()
        for q in search_queries:
            q_lower = q.lower().strip()
            if q_lower not in seen and len(q_lower) > 3:
                seen.add(q_lower)
                unique_queries.append(q)
        
        search_queries = unique_queries[:num_queries]
        
        logger.log(f"✅ Using {len(search_queries)} unique search queries:", "SUCCESS")
        for i, q in enumerate(search_queries, 1):
            logger.log(f"   {i}. {q}")
        
        # FIXED: Use user's max_results_per_search directly
        await self._send_websocket_progress(websocket, {
            "type": "status",
            "message": f"🔍 Searching {len(search_queries)} sources in parallel...",
            "stage": "parallel_search"
        })
        
        logger.log(f"\n🔍 PHASE 2: Parallel Google Search", "PROGRESS")
        logger.log(f"Executing {len(search_queries)} searches concurrently...")
        logger.log(f"Requesting {max_results_per_search} results per search (user specified)...")
        logger.log(f"Timeout: {config.SEARCH_TIMEOUT}s")
        
        # Run all Google searches concurrently with timeout
        search_tasks = [
            self.google_searcher.search(
                q, 
                num_results=max_results_per_search,  # FIXED: Use user's parameter
                session_id=session_id  # Pass session_id for caching
            )
            for q in search_queries
        ]
        
        try:
            all_results = await asyncio.wait_for(
                asyncio.gather(*search_tasks, return_exceptions=True),
                timeout=config.SEARCH_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.log("⚠️  Search timeout, using partial results", "WARNING")
            all_results = []
        
        # Deduplicate results
        logger.log(f"\n📊 PHASE 3: Results Processing", "PROGRESS")
        seen_urls = set()
        unique_results = []
        
        for i, results in enumerate(all_results, 1):
            if isinstance(results, list):
                logger.log(f"Search {i}: Found {len(results)} results")
                for r in results:
                    if r['link'] not in seen_urls:
                        seen_urls.add(r['link'])
                        unique_results.append(r)
            else:
                logger.log(f"Search {i}: Failed", "WARNING")
        
        logger.log(f"✅ Total unique URLs found: {len(unique_results)}", "SUCCESS")
        
        # FIXED: Calculate max URLs based on depth and max_results
        # More depth = more queries = potentially more URLs to scrape
        max_urls_to_scrape = min(
            len(unique_results),
            max_results_per_search * len(search_queries)  # Scale with queries
        )
        
        # But also cap at a reasonable maximum
        max_urls_to_scrape = min(max_urls_to_scrape, 30)  # Hard cap at 30
        
        top_urls = [r['link'] for r in unique_results[:max_urls_to_scrape]]
        
        await self._send_websocket_progress(websocket, {
            "type": "status",
            "message": f"📄 Extracting content from {len(top_urls)} pages...",
            "stage": "scraping"
        })
        
        logger.log(f"\n📄 PHASE 4: Content Extraction", "PROGRESS")
        logger.log(f"Scraping top {len(top_urls)} URLs with Jina Reader...")
        logger.log(f"Concurrent scrapes: {config.MAX_CONCURRENT_SCRAPES}")
        
        scraped_content = await self.web_scraper.scrape_multiple(top_urls, session_id=session_id)
        
        logger.log(f"✅ Successfully scraped {len(scraped_content)}/{len(top_urls)} pages", "SUCCESS")
        
        # Batch add all documents
        logger.log(f"\n💾 PHASE 5: Batch Knowledge Base Update", "PROGRESS")
        
        documents_to_add = []
        for result in unique_results:
            for scraped in scraped_content:
                if scraped['url'] == result['link']:
                    result['content'] = scraped['content']
                    documents_to_add.append({
                        'url': scraped['url'],
                        'title': result['title'],
                        'content': scraped['content'],
                        'query': query
                    })
                    all_search_results.append(result)
                    break
        
        if use_rag and documents_to_add:
            logger.log(f"🚀 Batch adding {len(documents_to_add)} documents (ONE embedding call)...")
            await vector_store.add_documents_batch(documents_to_add)
            logger.log(f"✅ Batch add complete!", "SUCCESS")
            logger.log(f"📚 Total documents in store: {vector_store.get_stats()['total_documents']}")
        
        # Use RAG to retrieve most relevant chunks
        await self._send_websocket_progress(websocket, {
            "type": "status",
            "message": f"🧠 Searching vector store for most relevant content...",
            "stage": "rag_retrieval"
        })
        
        logger.log(f"\n🧠 PHASE 6: RAG Retrieval", "PROGRESS")
        
        # FIXED: Scale chunks with depth
        max_chunks = min(config.MAX_CHUNKS_FOR_LLM, 10 + (depth * 5))  # More depth = more chunks
        
        if use_rag and vector_store.get_stats()['total_documents'] > 0:
            logger.log(f"🔍 Performing semantic search in vector store...")
            logger.log(f"   Query: '{query}'")
            logger.log(f"   Requesting top {max_chunks} most relevant chunks (scaled by depth)...")
            
            rag_results = await vector_store.search(query, n_results=max_chunks)
            
            logger.log(f"✅ Retrieved {len(rag_results)} relevant chunks from vector store", "SUCCESS")
            
            if rag_results:
                logger.log(f"   Similarity scores: {rag_results[0]['similarity_score']:.3f} (top) to {rag_results[-1]['similarity_score']:.3f} (lowest)")
            
            sources_for_llm = rag_results if rag_results else all_search_results[:max_chunks]
        else:
            logger.log(f"⚠️  No vector store available, using all scraped sources", "WARNING")
            sources_for_llm = all_search_results[:max_chunks]
        
        await self._send_websocket_progress(websocket, {
            "type": "status",
            "message": f"🧠 Analyzing {len(sources_for_llm)} most relevant chunks...",
            "stage": "analysis"
        })
        
        logger.log(f"\n🧠 PHASE 7: Answer Generation", "PROGRESS")
        logger.log(f"Sending {len(sources_for_llm)} most relevant sources to LLM...")
        logger.log(f"Total content size: ~{sum(len(s.get('content', '')) for s in sources_for_llm):,} chars")
        
        # Normalize URLs before passing to LLM
        for s in all_search_results:
            if not s.get("url") and s.get("link"):
                s["url"] = s["link"]

        raw_answer = await self.llm_client.generate_response(
            prompt=query,
            search_results=sources_for_llm
        )
        
        logger.log(f"✅ Generated raw answer: {len(raw_answer)} characters")

        # Format based on return type
        if return_format == "html":
            answer = format_answer_html(raw_answer, all_search_results)
            logger.log(f"✅ Formatted answer as HTML: {len(answer)} total chars (includes markup)")
        else:
            answer = format_answer_text(raw_answer)
            logger.log(f"✅ Formatted answer as plain text: {len(answer)} chars")
        
        # Store in session history
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
        
        self.session_histories[session_id].append({
            "query": query,
            "answer": answer,
            "sources": len(all_search_results),
            "timestamp": datetime.now().isoformat()
        })
        
        logger.log(f"💾 Saved to session history ({len(self.session_histories[session_id])} total)")
        
        # Prepare response
        cache_stats = self.cache.get_stats()
        vector_stats = vector_store.get_stats()
        
        result = {
            "query": query,
            "answer": answer,
            "session_id": session_id,
            "sources": [
                {
                    "title": r.get('title', 'No title'),
                    "url": r.get('url', r.get('link', '#')),
                    "snippet": r.get('content', r.get('snippet', ''))[:300]
                }
                for r in sources_for_llm[:10]
            ],
            "total_sources": len(all_search_results),
            "chunks_analyzed": len(sources_for_llm) if use_rag else 0,
            "search_depth": depth,
            "session_knowledge_size": vector_stats['total_documents'],
            "cache_stats": cache_stats
        }
        
        await self._send_websocket_progress(websocket, {
            "type": "complete",
            "message": "✅ Search complete!",
            "stage": "complete",
            "data": result
        })
        
        logger.log("\n" + "="*60)
        logger.log("✅ SEARCH COMPLETE!", "SUCCESS")
        logger.log("="*60)
        logger.log(f"📊 Results Summary:")
        logger.log(f"   • Total sources found: {result['total_sources']}")
        logger.log(f"   • Chunks analyzed by LLM: {result['chunks_analyzed']}")
        logger.log(f"   • Raw answer length: {len(raw_answer)} chars (LLM output)")
        logger.log(f"   • Formatted answer: {len(answer)} chars")
        logger.log(f"   • Format type: {return_format}")
        logger.log(f"   • Cache entries: {cache_stats['active_entries']}/{cache_stats['total_entries']}")
        logger.log(f"   • Vector DB docs: {vector_stats['total_documents']}")
        logger.log(f"   • Embedding provider: {vector_stats['embedding_provider']}")
        if use_rag:
            logger.log(f"   • RAG enabled: YES ✅ (used semantic search)")
        else:
            logger.log(f"   • RAG enabled: NO ❌ (direct scraping only)")
        elapsed_total = (datetime.now() - logger.start_time).total_seconds()
        logger.log(f"   • Total time: {elapsed_total:.2f}s")
        logger.log("="*60)
        
        result["steps_log"] = logger.get_steps()
        return result
    
    async def chat(
        self,
        message: str,
        session_id: str,
        use_search: bool = True,
        use_rag: bool = True,
        websocket: Optional[WebSocket] = None,
        return_format: str = "text"
    ) -> str:
        """Chat with session context"""
        print(f"\n💬 Starting chat in session {session_id[:12]}...")
        print(f"   Message: '{message[:60]}...'")
        print(f"   Use search: {use_search}")
        print(f"   Use RAG: {use_rag}")
        print(f"   Return format: {return_format}")
        
        context = self._build_session_context(session_id)
        
        if use_search:
            print("🔍 Triggering search for chat message...")
            result = await self.search(
                message,
                session_id=session_id,
                depth=1,
                max_results_per_search=5,
                use_rag=use_rag,
                websocket=websocket,
                return_format=return_format
            )
            print(f"✅ Chat search complete: {len(result['answer'])} chars")
            return result['answer']
        elif use_rag:
            vector_store = self.session_manager.get_or_create_session(session_id)
            
            if vector_store.get_stats()['total_documents'] > 0:
                print(f"📚 Searching session knowledge base...")
                await self._send_websocket_progress(websocket, {
                    "type": "status",
                    "message": "🧠 Searching session knowledge...",
                    "stage": "rag_search"
                })
                
                rag_results = await vector_store.search(message, n_results=5)
                print(f"✅ Found {len(rag_results)} relevant chunks")
                response = await self.llm_client.generate_response(
                    prompt=message,
                    context=context,
                    search_results=rag_results
                )
            else:
                print("⚠️  No documents in vector store, using context only")
                response = await self.llm_client.generate_response(
                    prompt=message,
                    context=context
                )
            
            # Format response
            if return_format == "text":
                response = format_answer_text(response)
            return response
        else:
            print("💭 Using context only (no search, no RAG)")
            response = await self.llm_client.generate_response(
                prompt=message,
                context=context
            )
            if return_format == "text":
                response = format_answer_text(response)
            return response
    
    def _build_session_context(self, session_id: str) -> str:
        """Build context from session's history"""
        if session_id not in self.session_histories:
            return ""
        
        history = self.session_histories[session_id]
        if not history:
            return ""
        
        context_parts = []
        for item in history[-5:]:
            context_parts.append(f"Q: {item['query']}")
            # Strip HTML for context
            clean_answer = format_answer_text(item['answer'])
            context_parts.append(f"A: {clean_answer[:800]}...")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a specific session"""
        vector_store = self.session_manager.get_or_create_session(session_id)
        stats = vector_store.get_stats()
        
        stats['session_id'] = session_id
        stats['search_history_length'] = len(self.session_histories.get(session_id, []))
        stats['active_sessions'] = self.session_manager.get_active_sessions_count()
        stats['llm_model'] = self.llm_client.get_model_info()
        stats['cache_stats'] = self.cache.get_stats()
        
        print(f"\n📊 Session Stats for {session_id[:12]}:")
        print(f"   • Documents: {stats['total_documents']}")
        print(f"   • Search history: {stats['search_history_length']} queries")
        print(f"   • Embedding provider: {stats['embedding_provider']}")
        
        return stats
    
    def get_knowledge_stats(self) -> Dict:
        """Get knowledge base statistics"""
        cache_stats = self.cache.get_stats()
        active_sessions = self.session_manager.get_active_sessions_count()
        
        print(f"\n📊 Knowledge Base Stats:")
        print(f"   • Active sessions: {active_sessions}")
        print(f"   • Cache entries: {cache_stats['active_entries']}/{cache_stats['total_entries']}")
        print(f"   • Cache size limit: {cache_stats['max_size']}")
        
        return {
            "cache_stats": cache_stats,
            "active_sessions": active_sessions
        }
    
    def end_session(self, session_id: str):
        """End a session and clean up its data"""
        print(f"\n👋 Ending session {session_id[:12]}...")
        
        # Delete vector store
        self.session_manager.delete_session(session_id)
        
        # Clear session cache
        asyncio.create_task(self.cache.clear_session(session_id))
        
        # Clear session history
        if session_id in self.session_histories:
            del self.session_histories[session_id]
        
        print(f"✅ Session {session_id[:12]}... ended and cleaned up (cache + vector store)")
    
    def clear_history(self):
        """Clear all session histories"""
        print("\n🗑️ Clearing all session histories...")
        self.session_histories = {}
        self.llm_client.clear_history()
        print("✅ All session histories cleared")