# src/search/jina_scraper.py - OPTIMIZED with Connection Pooling
import httpx
import asyncio
from typing import List, Dict, Optional

class JinaWebScraper:
    """
    FREE Jina AI Reader with Connection Pooling for 2x speed
    """
    
    def __init__(self):
        self.base_url = "https://r.jina.ai/"
        self.headers = {
            "X-Return-Format": "text"  # Get clean text instead of markdown
        }
        self.timeout = 15.0
        
        # Reuse HTTP client for connection pooling (MUCH FASTER!)
        self.client = None
    
    async def _get_client(self):
        """Get or create persistent HTTP client with connection pooling"""
        if self.client is None:
            self.client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_keepalive_connections=10,  # Keep connections alive
                    max_connections=20,
                    keepalive_expiry=30.0
                ),
                http2=True  # HTTP/2 for faster parallel requests
            )
            print("🔗 Created HTTP client with connection pooling")
        return self.client
    
    async def scrape_url(self, url: str) -> Optional[str]:
        """
        Scrape using FREE Jina Reader API with connection pooling
        """
        client = await self._get_client()
        
        try:
            # Jina Reader automatically extracts clean content
            jina_url = f"{self.base_url}{url}"
            
            response = await client.get(
                jina_url,
                headers=self.headers,
                follow_redirects=True
            )
            response.raise_for_status()
            
            content = response.text
            
            # Limit content length to avoid overwhelming the LLM
            if len(content) > 6000:
                content = content[:6000] + "... [truncated]"
            
            print(f"✅ Jina scraped {len(content)} chars from {url[:50]}...")
            return content
            
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP {e.response.status_code} for {url[:50]}...")
            return None
        except Exception as e:
            print(f"❌ Jina scrape failed for {url[:50]}...: {str(e)[:50]}")
            return None
    
    async def scrape_multiple(
        self, 
        urls: List[str], 
        max_concurrent: int = 10  # Increased from 5
    ) -> List[Dict[str, str]]:
        """
        Scrape multiple URLs with concurrency control
        Connection pooling makes this MUCH faster!
        """
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_limit(url: str):
            async with semaphore:
                content = await self.scrape_url(url)
                if content:
                    return {"url": url, "content": content}
                return None
        
        print(f"🔄 Scraping {len(urls)} URLs with Jina Reader (FREE, connection pooling enabled)...")
        
        # Execute with concurrency control
        tasks = [scrape_with_limit(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        print(f"✅ Jina scraped {len(valid_results)}/{len(urls)} websites")
        return valid_results
    
    async def close(self):
        """Close HTTP client and connections"""
        if self.client:
            await self.client.aclose()
            self.client = None
            print("🔌 Closed HTTP client connections")
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.client:
            try:
                asyncio.create_task(self.close())
            except:
                pass