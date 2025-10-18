from src.search.google_search import GoogleSearcher
from src.search.jina_scraper import JinaWebScraper

class SimpleCache:
    
    def __init__(self):

        self.search_cache = {}
        self.scrape_cache = {}
        print("Cache initialized")
    
    def get_search(self, query: str):
        """Check if we've searched for this before"""
        result = self.search_cache.get(query)
        
        if result:
            print(f"Cache HIT for search: {query[:50]}...")
        
        return result
    
    def save_search(self, query: str, results: list):
        self.search_cache[query] = results
        print(f"Cached search for: {query[:50]}...")
    
    def get_scrape(self, url: str):
        """Check if we've scraped this URL before"""
        return self.scrape_cache.get(url)
    
    def save_scrape(self, url: str, content: str):
        self.scrape_cache[url] = content
    
    def clear(self):
        self.search_cache.clear()
        self.scrape_cache.clear()
        print("🗑️ Cache cleared")

class CachedGoogleSearcher:
    
    def __init__(self, api_key: str, cse_id: str, cache: SimpleCache):
        self.searcher = GoogleSearcher(api_key, cse_id)
        self.cache = cache
    
    async def search(self, query: str, num_results: int = 10):
        """Search with caching"""
        cached = self.cache.get_search(query)
        if cached:
            return cached[:num_results]
        
        results = await self.searcher.search(query, num_results)

        if results:
            self.cache.save_search(query, results)
        
        return results

class CachedJinaScraper:
    
    def __init__(self, cache: SimpleCache):

        self.scraper = JinaWebScraper()
        self.cache = cache
    
    async def scrape_url(self, url: str):
        """Scrape a single URL with caching"""
        cached = self.cache.get_scrape(url)
        if cached:
            return cached
        
        content = await self.scraper.scrape_url(url)
        
        if content:
            self.cache.save_scrape(url, content)
        
        return content
    
    async def scrape_multiple(self, urls: list):
        """Scrape multiple URLs with caching"""
        results = []
        urls_to_scrape = []
        
        for url in urls:
            cached = self.cache.get_scrape(url)
            if cached:
                results.append({"url": url, "content": cached})
            else:
                urls_to_scrape.append(url)
        
        if urls_to_scrape:
            print(f"Scraping {len(urls_to_scrape)} new URLs (rest from cache)...")
            scraped = await self.scraper.scrape_multiple(urls_to_scrape)
            
            for item in scraped:
                self.cache.save_scrape(item['url'], item['content'])
                results.append(item)
        
        print(f"Total: {len(results)} URLs ({len(results) - len(urls_to_scrape)} from cache)")
        return results