import asyncio
from typing import List, Dict, Optional
from datetime import datetime
from fastapi import WebSocket
import html
import re

from src.config import config
from src.cache.memory_cache import SimpleCache, CachedGoogleSearcher, CachedJinaScraper
from src.llm.openrouter_client import OpenRouterClient
from src.rag.vector_store import VectorStore

BASE_CHUNKS_TO_RETRIEVE = 10
CHUNKS_PER_DEPTH_LEVEL = 3
MAX_CHUNKS_TO_RETRIEVE = 25
MAX_PAGES_TO_SCRAPE = 25

class DeepSearchPipeline:
    
    def __init__(self):
        print("\nInitializing Deep Search Pipeline...")
        
        config.validate()
        
        print("Setting up cache...")
        self.cache = SimpleCache()

        print("Setting up Google Search...")
        self.google_searcher = CachedGoogleSearcher(config.GOOGLE_SEARCH_API_KEY,
                                                    config.GOOGLE_CSE_ID,
                                                    self.cache)
        
        print("Setting up web scraper...")
        self.web_scraper = CachedJinaScraper(self.cache)

        print("Setting up LLM...")
        self.llm_client = OpenRouterClient(config.OPENROUTER_API_KEY,
                                           model_name=config.OPENROUTER_MODEL)
        
        print("Setting up vector store...")
        self.vector_store = VectorStore()
        
        print("--- Deep Search Pipeline ready! ---\n")
    
    async def _send_update(self, websocket: Optional[WebSocket], message: str, stage: str = "progress"):
        """Send a progress update via WebSocket"""
        if not websocket:
            return
        
        try:
            await websocket.send_json({"type": "status",
                                       "message": message,
                                       "stage": stage})
        except Exception as e:
            pass

    def _deduplicate_queries(self, queries: List[str], original_query: str) -> List[str]:
        """
        Remove duplicate queries and ensure the original query is included.
        
        Args:
            queries: List of generated search queries
            original_query: User's question
        Returns:
            Deduplicated list with user's query
        """
        queries.insert(0, original_query)
        
        unique_queries = []
        seen_lowercase = set()
        
        for query in queries:
            query_lower = query.lower().strip()

            if query_lower not in seen_lowercase:
                seen_lowercase.add(query_lower)
                unique_queries.append(query)
        
        return unique_queries

    def _prepare_documents(self, scraped_data: List[Dict], url_metadata: Dict) -> List[Dict]:
        """        
        Combines the scraped text content with metadata
        
        Args:
            scraped_data: List of dicts with scraped content:
                [{"url": str, "content": str}, ...]
            url_metadata: Dict mapping URLs to their search result metadata:
                {url1: {"title": str, "snippet": str, ...}, ...}
        Returns:
            List of document dicts ready for vector store:
                [{"url": str, "title": str, "content": str, "query": str}, ...]
        """
        documents = []
        
        for scraped in scraped_data:
            url = scraped['url']
            content = scraped['content']
            
            metadata = url_metadata.get(url, {})
            title = metadata.get('title', 'Untitled')
            documents.append({'url': url,
                              'title': title,
                              'content': content,
                              'query': metadata.get('query', '')})
        
        return documents

    def _format_answer_html(self, text: str, sources: List[Dict]) -> str:
        """Format AI answer text into HTML with clickable source links"""
        url_map = {}
        source_titles = {}
        used_sources = set()

        for i, source in enumerate(sources, 1):
            url_map[i] = source.get('url', '#')
            source_titles[i] = source.get('title', f'Source {i}')
        
        text = html.escape(text)
        
        # "Source 1, 2, 3" or "(Source 4, 5)"
        pattern1 = r'\(?Source\s+(\d+(?:\s*,\s*\d+)*)\)?'
        matches = re.finditer(pattern1, text, re.IGNORECASE)
        
        for match in reversed(list(matches)):
            full_text = match.group(0)
            numbers_part = match.group(1)
            
            nums = [int(n) for n in re.findall(r'\d+', numbers_part)]
            
            links = []
            for num in nums:

                if num in url_map:
                    used_sources.add(num)
                    links.append(f'<a href="{url_map[num]}" target="_blank" class="source-link">Source {num}</a>')
                else:
                    links.append(f'Source {num}')
            
            if full_text.startswith('('):
                replacement = f'({", ".join(links)})'
            else:
                replacement = ', '.join(links)
            
            start, end = match.span()
            text = text[:start] + replacement + text[end:]
        
        # "Source N"
        pattern2 = r'Source\s+(\d+)'
        
        def make_link(match):
            num = int(match.group(1))

            if num in url_map:
                used_sources.add(num)
                return f'<a href="{url_map[num]}" target="_blank" class="source-link">Source {num}</a>'
            
            return match.group(0)
        
        text = re.sub(pattern2, make_link, text, flags=re.IGNORECASE)
        
        result_html = '<div class="chat-block">'
        
        paragraphs = text.split('\n\n')

        for para in paragraphs:
            if para.strip():
                para = para.replace('\n', '<br>')
                result_html += f'<p>{para}</p>'

        if used_sources:
            result_html += '<div class="sources-section">'
            result_html += '<div class="sources-header">Sources:</div>'
            
            for num in sorted(used_sources):
                title = source_titles[num]
                if len(title) > 80:
                    title = title[:80] + '...'
                
                result_html += f'<div class="source-item">'
                result_html += f'<span class="source-number">[{num}]</span> '
                result_html += f'<a href="{html.escape(url_map[num])}" target="_blank" class="source-link">{html.escape(title)}</a>'
                result_html += '</div>'
            
            result_html += '</div>'
        
        result_html += '</div>'
        return result_html
    
    async def search(self, 
                     query: str, 
                     depth: int = 2, 
                     max_results: int = 7, 
                     websocket: Optional[WebSocket] = None
                     ) -> Dict:
        """Perform a deep search for the given query."""
        start_time = datetime.now()
        print(f"Starting search for: '{query}'")
        print(f"Depth: {depth} | Max results: {max_results}")
        
        await self._send_update(websocket, "Starting deep search...", "starting")
        
        await self._send_update(websocket, "Generating search queries...", "planning")
        
        print(f"Generating {depth + 1} search queries...")
        num_queries = depth + 1
        generated_queries = await self.llm_client.generate_search_queries(query, num_queries)
        search_queries = self._deduplicate_queries(generated_queries, query)
        search_queries = search_queries[:num_queries]
        
        print(f"Using {len(search_queries)} queries:")
        for i, q in enumerate(search_queries, 1):
            print(f"{i}. {q}")
        
        await self._send_update(websocket, 
                                f"Searching Google with {len(search_queries)} queries...", 
                                "searching")
        
        print(f"\nSearching Google ({len(search_queries)} queries)...")
        
        search_tasks = [self.google_searcher.search(q, num_results = max_results) for q in search_queries]
        
        all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        unique_urls = []
        url_metadata  = {}
        for results in all_results:

            if isinstance(results, list):
                for result in results:

                    url = result.get('link')
                    if url and url not in url_metadata:

                        url_metadata[url] = result
                        unique_urls.append(url)
        
        print(f"Found {len(unique_urls)} unique URLs")
        
        pages_to_scrape = min(len(unique_urls), max_results * len(search_queries), MAX_PAGES_TO_SCRAPE)
        urls_to_scrape = unique_urls[:pages_to_scrape]
        
        await self._send_update(websocket, f"Scraping {len(urls_to_scrape)} pages...", "scraping")
        
        print(f"\nScraping {len(urls_to_scrape)} pages...")
        scraped_data = await self.web_scraper.scrape_multiple(urls_to_scrape)
        print(f"Successfully scraped {len(scraped_data)} pages")
 
        await self._send_update(websocket, f"Adding {len(scraped_data)} documents to vector database...", "indexing")
        print(f"\nAdding to vector database...")

        documents = self._prepare_documents(scraped_data, url_metadata)
        if documents:
            await self.vector_store.add_documents_batch(documents)
            print(f"Added {len(documents)} documents to vector store")
        
        stats = self.vector_store.get_stats()
        print(f"Total documents in store: {stats['total_documents']}")

        await self._send_update(websocket, "Finding most relevant information...", "analyzing")
        
        print(f"\nFinding most relevant content...")
        
        num_chunks = min(
            BASE_CHUNKS_TO_RETRIEVE + (depth * CHUNKS_PER_DEPTH_LEVEL),
            MAX_CHUNKS_TO_RETRIEVE
        )
        relevant_chunks = await self.vector_store.search(query, n_results = num_chunks)
        
        print(f"Found {len(relevant_chunks)} relevant chunks")
        print(f"Similarity range: {relevant_chunks[0]['similarity_score']:.3f} to {relevant_chunks[-1]['similarity_score']:.3f}")
        
        await self._send_update(websocket, "Generating comprehensive answer...", "generating")
        
        print(f"\nGenerating answer with LLM...")

        answer = await self.llm_client.generate_response(prompt = query,
                                                         search_results = relevant_chunks)
        
        print(f"Generated answer ({len(answer)} chars)")

        formatted_answer = self._format_answer_html(answer, relevant_chunks)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        result = {"query": query,
                  "answer": formatted_answer,
                  "total_sources": len(scraped_data),
                  "chunks_analyzed": len(relevant_chunks),
                  "time_seconds": elapsed,
                  "sources": [{"title": chunk.get('title', 'Untitled'),
                               "url": chunk.get('url', '#'),
                               "snippet": chunk.get('content', '')[:200]} for chunk in relevant_chunks[:10]]}
        
        await self._send_update(websocket, "Search complete!", "complete")
        
        print(f"Search complete in {elapsed:.2f}s")
        
        return result
    
    def clear_knowledge(self):
        print("Clearing vector store...")
        self.vector_store = VectorStore()
        print("Vector store cleared")