# src/search/jina_scraper.py
import httpx
import asyncio
from typing import List, Dict, Optional

class JinaWebScraper:
    """
    FREE Jina AI Reader - 1000 requests/day free tier
    Much faster than custom HTML parsing
    No API key needed!
    """
    
    def __init__(self):
        self.base_url = "https://r.jina.ai/"
        self.headers = {
            "X-Return-Format": "text"  # Get clean text instead of markdown
        }
        self.timeout = 15.0
    
    async def scrape_url(self, url: str) -> Optional[str]:
        """
        Scrape using FREE Jina Reader API
        Returns clean text content
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # Jina Reader automatically extracts clean content
                # Format: https://r.jina.ai/{original_url}
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
        max_concurrent: int = 5  # Limit to avoid rate limits
    ) -> List[Dict[str, str]]:
        """
        Scrape multiple URLs with concurrency control
        """
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_limit(url: str):
            async with semaphore:
                content = await self.scrape_url(url)
                if content:
                    return {"url": url, "content": content}
                return None
        
        print(f"🔄 Scraping {len(urls)} URLs with Jina Reader (FREE)...")
        
        # Execute with concurrency control
        tasks = [scrape_with_limit(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        print(f"✅ Jina scraped {len(valid_results)}/{len(urls)} websites")
        return valid_results