"""
Jina AI Reader - Free web scraping service

Instead of parsing HTML ourselves, we just send URLs to Jina
and they send back clean text. Much easier than dealing with
all the messy HTML parsing ourselves.
"""

import httpx
import asyncio
from typing import List, Dict, Optional


class JinaWebScraper:
    
    def __init__(self):
        self.base_url = "https://r.jina.ai/"
        self.timeout = 15.0
    
    async def scrape_url(self, url: str) -> Optional[str]:
        """      
        Returns clean text or None if it fails
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                jina_url = f"{self.base_url}{url}"
                
                response = await client.get(jina_url, follow_redirects=True)
                response.raise_for_status()
                
                content = response.text
                
                if len(content) > 6000:
                    content = content[:6000] + "..."
                
                print(f"✅ Scraped {len(content)} chars from {url[:50]}...")
                return content
                
            except httpx.HTTPStatusError as e:
                print(f"❌ HTTP {e.response.status_code} for {url[:50]}...")
                return None
            except Exception as e:
                print(f"❌ Failed to scrape {url[:50]}...: {str(e)[:50]}")
                return None
    
    async def scrape_multiple(self, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, str]]:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_one(url: str):
            async with semaphore:
                content = await self.scrape_url(url)
                if content:
                    return {"url": url, "content": content}
                return None
        
        print(f"🔄 Scraping {len(urls)} URLs (max {max_concurrent} at once)...")
        
        tasks = [scrape_one(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        valid_results = [r for r in results if r is not None]
        
        print(f"✅ Successfully scraped {len(valid_results)}/{len(urls)} URLs")
        return valid_results