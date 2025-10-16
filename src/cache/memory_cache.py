# src/cache/memory_cache.py - FIXED: Session-based cache only
import hashlib
import time
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
import asyncio

class MemoryCache:
    """
    Session-based in-memory cache
    Cache is cleared when session ends - NO persistence!
    """
    
    def __init__(self, default_ttl_hours: int = 24, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.max_size = max_size
        self.session_caches: Dict[str, Dict] = {}  # session_id -> cache dict
        
        print(f"✅ Session-based cache initialized (max {max_size} entries per session)")
        
        # Start background cleanup
        asyncio.create_task(self._cleanup_expired())
    
    def _make_key(self, prefix: str, data: str) -> str:
        """Create cache key from data"""
        hash_val = hashlib.md5(data.encode()).hexdigest()
        return f"{prefix}:{hash_val}"
    
    def _is_expired(self, entry: Dict) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() > entry['expires_at']
    
    def _evict_oldest(self, session_id: str = None):
        """Remove oldest entries if cache is too large"""
        cache_to_check = self.session_caches.get(session_id, {}) if session_id else self.cache
        
        if len(cache_to_check) > self.max_size:
            # Sort by creation time and remove oldest
            sorted_keys = sorted(
                cache_to_check.keys(),
                key=lambda k: cache_to_check[k]['created_at']
            )
            
            # Remove oldest 10%
            num_to_remove = max(1, len(cache_to_check) // 10)
            for key in sorted_keys[:num_to_remove]:
                del cache_to_check[key]
            
            print(f"🗑️ Evicted {num_to_remove} old cache entries")
    
    async def _cleanup_expired(self):
        """Background task to clean up expired entries"""
        while True:
            await asyncio.sleep(3600)  # Run every hour
            
            # Clean global cache
            expired_keys = [
                key for key, entry in self.cache.items()
                if self._is_expired(entry)
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            # Clean session caches
            for session_id, session_cache in self.session_caches.items():
                expired_keys = [
                    key for key, entry in session_cache.items()
                    if self._is_expired(entry)
                ]
                for key in expired_keys:
                    del session_cache[key]
            
            if expired_keys:
                print(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
    
    async def get_search_results(self, query: str, session_id: str = None) -> Optional[list]:
        """Get cached search results"""
        key = self._make_key("search", query)
        cache_to_use = self.session_caches.get(session_id, {}) if session_id else self.cache
        
        if key in cache_to_use:
            entry = cache_to_use[key]
            
            if not self._is_expired(entry):
                print(f"✅ Cache HIT for search: {query[:50]}...")
                return entry['value']
            else:
                # Remove expired entry
                del cache_to_use[key]
        
        print(f"❌ Cache MISS for search: {query[:50]}...")
        return None
    
    async def set_search_results(self, query: str, results: list, session_id: str = None):
        """Cache search results"""
        key = self._make_key("search", query)
        
        # Get or create session cache
        if session_id:
            if session_id not in self.session_caches:
                self.session_caches[session_id] = {}
            cache_to_use = self.session_caches[session_id]
        else:
            cache_to_use = self.cache
        
        cache_to_use[key] = {
            'value': results,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + self.default_ttl
        }
        
        self._evict_oldest(session_id)
        print(f"💾 Cached search results for: {query[:50]}...")
    
    async def get_scraped_content(self, url: str, session_id: str = None) -> Optional[str]:
        """Get cached scraped content"""
        key = self._make_key("scrape", url)
        cache_to_use = self.session_caches.get(session_id, {}) if session_id else self.cache
        
        if key in cache_to_use:
            entry = cache_to_use[key]
            
            if not self._is_expired(entry):
                print(f"✅ Cache HIT for URL: {url[:50]}...")
                return entry['value']
            else:
                del cache_to_use[key]
        
        return None
    
    async def set_scraped_content(self, url: str, content: str, session_id: str = None):
        """Cache scraped content"""
        key = self._make_key("scrape", url)
        
        # Get or create session cache
        if session_id:
            if session_id not in self.session_caches:
                self.session_caches[session_id] = {}
            cache_to_use = self.session_caches[session_id]
        else:
            cache_to_use = self.cache
        
        cache_to_use[key] = {
            'value': content,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(hours=24)  # 24h TTL for scrapes
        }
        
        self._evict_oldest(session_id)
    
    async def clear_session(self, session_id: str):
        """Clear cache for a specific session"""
        if session_id in self.session_caches:
            count = len(self.session_caches[session_id])
            del self.session_caches[session_id]
            print(f"🗑️ Cleared {count} cache entries for session {session_id[:12]}...")
    
    async def clear_all(self):
        """Clear all cache"""
        global_count = len(self.cache)
        session_count = sum(len(sc) for sc in self.session_caches.values())
        
        self.cache.clear()
        self.session_caches.clear()
        
        print(f"🗑️ Memory cache cleared: {global_count} global + {session_count} session entries")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        global_total = len(self.cache)
        global_expired = sum(1 for entry in self.cache.values() if self._is_expired(entry))
        
        session_total = sum(len(sc) for sc in self.session_caches.values())
        session_expired = sum(
            sum(1 for entry in sc.values() if self._is_expired(entry))
            for sc in self.session_caches.values()
        )
        
        return {
            "total_entries": global_total + session_total,
            "global_entries": global_total,
            "session_entries": session_total,
            "expired_entries": global_expired + session_expired,
            "active_entries": (global_total - global_expired) + (session_total - session_expired),
            "active_sessions": len(self.session_caches),
            "max_size": self.max_size
        }


# Cached wrapper for GoogleSearcher
class CachedGoogleSearcher:
    """Google Search with session-based caching"""
    
    def __init__(self, api_key: str, cse_id: str, cache: MemoryCache):
        from src.search.google_search import GoogleSearcher
        self.searcher = GoogleSearcher(api_key, cse_id)
        self.cache = cache
    
    async def search(self, query: str, num_results: int = 10, session_id: str = None) -> list:
        # Check cache first
        cached = await self.cache.get_search_results(query, session_id)
        if cached:
            return cached[:num_results]
        
        # If not cached, do actual search
        results = await self.searcher.search(query, num_results)
        
        # Cache the results
        if results:
            await self.cache.set_search_results(query, results, session_id)
        
        return results


# Cached wrapper for JinaWebScraper
class CachedJinaScraper:
    """Jina scraper with session-based caching"""
    
    def __init__(self, cache: MemoryCache):
        from src.search.jina_scraper import JinaWebScraper
        self.scraper = JinaWebScraper()
        self.cache = cache
    
    async def scrape_url(self, url: str, session_id: str = None) -> Optional[str]:
        # Check cache first
        cached = await self.cache.get_scraped_content(url, session_id)
        if cached:
            return cached
        
        # If not cached, scrape
        content = await self.scraper.scrape_url(url)
        
        # Cache the content
        if content:
            await self.cache.set_scraped_content(url, content, session_id)
        
        return content
    
    async def scrape_multiple(self, urls: list, session_id: str = None) -> list:
        results = []
        uncached_urls = []
        
        # Check cache for all URLs
        for url in urls:
            cached = await self.cache.get_scraped_content(url, session_id)
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
                await self.cache.set_scraped_content(item['url'], item['content'], session_id)
                results.append(item)
        
        print(f"📊 Total: {len(results)} URLs ({len(results) - len(uncached_urls)} from cache)")
        return results