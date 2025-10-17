"""
Deep Search Pipeline
Simplified to focus on core functionality
"""

import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from fastapi import WebSocket

from src.config import config
from src.cache.memory_cache import SimpleCache, CachedGoogleSearcher, CachedJinaScraper
from src.llm.openrouter_client import OpenRouterClient
from src.rag.vector_store import VectorStore


class DeepSearchPipeline:
    """
    Core deep search pipeline
    Takes a query, searches Google, scrapes pages, stores in vector DB, and generates answer
    """
    
    def __init__(self):
        print("\n🚀 Initializing Deep Search Pipeline...")
        
        config.validate()
        
        print("📦 Setting up cache...")
        self.cache = SimpleCache()

        print("🔍 Setting up Google Search...")
        self.google_searcher = CachedGoogleSearcher(
            config.GOOGLE_SEARCH_API_KEY,
            config.GOOGLE_CSE_ID,
            self.cache
        )
        
        print("🌐 Setting up web scraper...")
        self.web_scraper = CachedJinaScraper(self.cache)

        print("🤖 Setting up LLM...")
        self.llm_client = OpenRouterClient(
            config.OPENROUTER_API_KEY,
            model_name=config.OPENROUTER_MODEL
        )
        
        print("💾 Setting up vector store...")
        self.vector_store = VectorStore()
        
        print("✅ Deep Search Pipeline ready!\n")
    
    async def _send_update(self, websocket: Optional[WebSocket], message: str, stage: str = "progress"):
        if not websocket:
            return
        
        try:
            await websocket.send_json({
                "type": "status",
                "message": message,
                "stage": stage
            })
        except Exception as e:
            pass
    
    async def search(
        self,
        query: str,
        depth: int = 2,
        max_results: int = 7,
        websocket: Optional[WebSocket] = None
    ) -> Dict:
        """
        Main search function
        
        Steps:
        1. Generate multiple search queries using AI (based on depth)
        2. Search Google with all those queries in parallel
        3. Scrape the top results
        4. Store everything in vector database
        5. Find most relevant chunks using semantic search
        6. Send those chunks to LLM to generate final answer
        
        Args:
            query: What the user wants to know
            depth: How many search queries to generate (1-5)
            max_results: How many Google results per query (1-15)
            websocket: Optional WebSocket connection for real-time progress updates
        """
        start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"🔍 Starting search for: '{query}'")
        print(f"   Depth: {depth} | Max results: {max_results}")
        print(f"{'='*60}\n")
        
        await self._send_update(websocket, "🚀 Starting deep search...", "starting")
        
        await self._send_update(websocket, "🧠 Generating search queries...", "planning")
        
        print(f"🧠 Generating {depth + 1} search queries...")
        num_queries = depth + 1
        search_queries = await self.llm_client.generate_search_queries(query, num_queries)
        
        if query not in search_queries:
            search_queries.insert(0, query)
        
        unique_queries = []
        seen = set()
        for q in search_queries:
            q_lower = q.lower().strip()
            if q_lower not in seen:
                seen.add(q_lower)
                unique_queries.append(q)
        
        search_queries = unique_queries[:num_queries]
        
        print(f"✅ Using {len(search_queries)} queries:")
        for i, q in enumerate(search_queries, 1):
            print(f"   {i}. {q}")
        
        await self._send_update(
            websocket, 
            f"🔍 Searching Google with {len(search_queries)} queries...", 
            "searching"
        )
        
        print(f"\n🔍 Searching Google ({len(search_queries)} queries in parallel)...")
        
        search_tasks = [
            self.google_searcher.search(q, num_results=max_results)
            for q in search_queries
        ]
        
        all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        all_urls = []
        url_to_result = {}
        
        for results in all_results:
            if isinstance(results, list):
                for r in results:
                    url = r.get('link')
                    if url and url not in url_to_result:
                        url_to_result[url] = r
                        all_urls.append(url)
        
        print(f"✅ Found {len(all_urls)} unique URLs")
        
        max_to_scrape = min(len(all_urls), max_results * len(search_queries), 25)
        urls_to_scrape = all_urls[:max_to_scrape]
        
        await self._send_update(
            websocket,
            f"📄 Scraping {len(urls_to_scrape)} pages...",
            "scraping"
        )
        
        print(f"\n📄 Scraping {len(urls_to_scrape)} pages...")
        scraped_data = await self.web_scraper.scrape_multiple(urls_to_scrape)
        print(f"✅ Successfully scraped {len(scraped_data)} pages")
 
        await self._send_update(
            websocket,
            f"💾 Adding {len(scraped_data)} documents to knowledge base...",
            "indexing"
        )
        
        print(f"\n💾 Adding to vector database...")

        documents_to_add = []
        for scraped in scraped_data:
            url = scraped['url']
            content = scraped['content']
            
            result = url_to_result.get(url, {})
            title = result.get('title', 'Untitled')
            
            documents_to_add.append({
                'url': url,
                'title': title,
                'content': content,
                'query': query
            })
        
        if documents_to_add:
            await self.vector_store.add_documents_batch(documents_to_add)
            print(f"✅ Added {len(documents_to_add)} documents to vector store")
        
        print(f"📚 Total documents in store: {self.vector_store.get_stats()['total_documents']}")

        await self._send_update(
            websocket,
            "🧠 Finding most relevant information...",
            "analyzing"
        )
        
        print(f"\n🧠 Finding most relevant content...")
        
        num_chunks = min(10 + (depth * 3), 25)
        relevant_chunks = await self.vector_store.search(query, n_results=num_chunks)
        
        print(f"✅ Found {len(relevant_chunks)} relevant chunks")
        if relevant_chunks:
            top_score = relevant_chunks[0]['similarity_score']
            lowest_score = relevant_chunks[-1]['similarity_score']
            print(f"   Similarity range: {top_score:.3f} to {lowest_score:.3f}")
        
        await self._send_update(
            websocket,
            "🤖 Generating comprehensive answer...",
            "generating"
        )
        
        print(f"\n🤖 Generating answer with LLM...")
        answer = await self.llm_client.generate_response(
            prompt=query,
            search_results=relevant_chunks
        )
        
        print(f"✅ Generated answer ({len(answer)} chars)")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        result = {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "title": chunk.get('title', 'Untitled'),
                    "url": chunk.get('url', '#'),
                    "snippet": chunk.get('content', '')[:200]
                }
                for chunk in relevant_chunks[:10]  # Top 10 sources
            ],
            "total_sources": len(scraped_data),
            "chunks_analyzed": len(relevant_chunks),
            "time_seconds": elapsed
        }
        
        await self._send_update(
            websocket,
            "✅ Search complete!",
            "complete"
        )
        
        print(f"\n{'='*60}")
        print(f"✅ Search complete in {elapsed:.2f}s")
        print(f"{'='*60}\n")
        
        return result
    
    def clear_knowledge(self):
        print("🗑️ Clearing vector store...")
        self.vector_store = VectorStore()
        print("✅ Vector store cleared")