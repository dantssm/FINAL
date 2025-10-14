# src/cache/memory_cache.py
import hashlib
import time
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
import asyncio

class MemoryCache:
    """
    FREE in-memory cache (no Redis needed)
    Stores search results and scraped content in RAM
    Perfect for single-server deployments
    """
    
    def __init__(self, default_ttl_hours: int = 24, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.max_size = max_size
        
        # Start background cleanup task
        asyncio.create_task(self._cleanup_expired())
    
    def _make_key(self, prefix: str, data: str) -> str:
        """Create cache key from data"""
        hash_val = hashlib.md5(data.encode()).hexdigest()
        return f"{prefix}:{hash_val}"
    
    def _is_expired(self, entry: Dict) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() > entry['expires_at']
    
    def _evict_oldest(self):
        """Remove oldest entries if cache is too large"""
        if len(self.cache) > self.max_size:
            # Sort by creation time and remove oldest
            sorted_keys = sorted(
                self.cache.keys(),
                key=lambda k: self.cache[k]['created_at']
            )
            
            # Remove oldest 10%
            num_to_remove = max(1, len(self.cache) // 10)
            for key in sorted_keys[:num_to_remove]:
                del self.cache[key]
            
            print(f"🗑️ Evicted {num_to_remove} old cache entries")
    
    async def _cleanup_expired(self):
        """Background task to clean up expired entries"""
        while True:
            await asyncio.sleep(3600)  # Run every hour
            
            expired_keys = [
                key for key, entry in self.cache.items()
                if self._is_expired(entry)
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            if expired_keys:
                print(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
    
    async def get_search_results(self, query: str) -> Optional[list]:
        """Get cached search results"""
        key = self._make_key("search", query)
        
        if key in self.cache:
            entry = self.cache[key]
            
            if not self._is_expired(entry):
                print(f"✅ Cache HIT for search: {query[:50]}...")
                return entry['value']
            else:
                # Remove expired entry
                del self.cache[key]
        
        print(f"❌ Cache MISS for search: {query[:50]}...")
        return None
    
    async def set_search_results(self, query: str, results: list):
        """Cache search results"""
        key = self._make_key("search", query)
        
        self.cache[key] = {
            'value': results,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + self.default_ttl
        }
        
        self._evict_oldest()
        print(f"💾 Cached search results for: {query[:50]}...")
    
    async def get_scraped_content(self, url: str) -> Optional[str]:
        """Get cached scraped content"""
        key = self._make_key("scrape", url)
        
        if key in self.cache:
            entry = self.cache[key]
            
            if not self._is_expired(entry):
                print(f"✅ Cache HIT for URL: {url[:50]}...")
                return entry['value']
            else:
                del self.cache[key]
        
        return None
    
    async def set_scraped_content(self, url: str, content: str):
        """Cache scraped content (longer TTL)"""
        key = self._make_key("scrape", url)
        
        self.cache[key] = {
            'value': content,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(days=7)  # Cache for 7 days
        }
        
        self._evict_oldest()
    
    async def clear_all(self):
        """Clear all cache"""
        self.cache.clear()
        print("🗑️ Memory cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = len(self.cache)
        expired = sum(1 for entry in self.cache.values() if self._is_expired(entry))
        
        return {
            "total_entries": total,
            "expired_entries": expired,
            "active_entries": total - expired,
            "max_size": self.max_size
        }


# Cached wrapper for GoogleSearcher
class CachedGoogleSearcher:
    """Google Search with FREE in-memory caching"""
    
    def __init__(self, api_key: str, cse_id: str, cache: MemoryCache):
        from src.search.google_search import GoogleSearcher
        self.searcher = GoogleSearcher(api_key, cse_id)
        self.cache = cache
    
    async def search(self, query: str, num_results: int = 10) -> list:
        # Check cache first
        cached = await self.cache.get_search_results(query)
        if cached:
            return cached[:num_results]
        
        # If not cached, do actual search
        results = await self.searcher.search(query, num_results)
        
        # Cache the results
        if results:
            await self.cache.set_search_results(query, results)
        
        return results


# Cached wrapper for JinaWebScraper
class CachedJinaScraper:
    """Jina scraper with FREE in-memory caching"""
    
    def __init__(self, cache: MemoryCache):
        from src.search.jina_scraper import JinaWebScraper
        self.scraper = JinaWebScraper()
        self.cache = cache
    
    async def scrape_url(self, url: str) -> Optional[str]:
        # Check cache first
        cached = await self.cache.get_scraped_content(url)
        if cached:
            return cached
        
        # If not cached, scrape
        content = await self.scraper.scrape_url(url)
        
        # Cache the content
        if content:
            await self.cache.set_scraped_content(url, content)
        
        return content
    
    async def scrape_multiple(self, urls: list) -> list:
        results = []
        uncached_urls = []
        
        # Check cache for all URLs
        for url in urls:
            cached = await self.cache.get_scraped_content(url)
            if cached:
                results.append({"url": url, "content": cached})
            else:
                uncached_urls.append(url)
        
        # Scrape uncached URLs
        if uncached_urls:
            print(f"🔄 Scraping {len(uncached_urls)} uncached URLs...")
            scraped = await self.scraper.scrape_multiple(uncached_urls)
            
            # Cache newly scraped content
            for item in scraped:
                await self.cache.set_scraped_content(item['url'], item['content'])
                results.append(item)
        
        print(f"📊 Total: {len(results)} URLs ({len(results) - len(uncached_urls)} from cache)")
        return results