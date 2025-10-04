# src/search/web_scraper.py
import httpx
import asyncio
from typing import Optional, List, Dict
from selectolax.parser import HTMLParser

class WebScraper:
    def __init__(self):
        # Headers to look like a real browser
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
    async def scrape_url(self, url: str) -> Optional[str]:
        """
        Scrape content from a single URL
        
        Args:
            url: Website URL to scrape
            
        Returns:
            Extracted text content or None if failed
        """
        async with httpx.AsyncClient() as client:
            try:
                # Fetch the page
                response = await client.get(
                    url,
                    headers=self.headers,
                    timeout=10.0,
                    follow_redirects=True
                )
                response.raise_for_status()
                
                # Parse HTML
                parser = HTMLParser(response.text)
                
                # Remove script and style elements
                for tag in parser.css('script, style, nav, header, footer'):
                    tag.decompose()
                
                # Get text content
                if parser.body:
                    text = parser.body.text(separator=' ', strip=True)
                    
                    # Clean up text
                    text = ' '.join(text.split())  # Remove extra whitespace
                    
                    # Limit length (for now, 5000 chars)
                    text = text[:5000]
                    
                    print(f"✅ Scraped {len(text)} chars from {url[:50]}...")
                    return text
                else:
                    print(f"⚠️ No content found at {url[:50]}...")
                    return None
                    
            except httpx.HTTPStatusError as e:
                print(f"❌ HTTP error {e.response.status_code} for {url[:50]}...")
                return None
            except Exception as e:
                print(f"❌ Failed to scrape {url[:50]}...: {str(e)[:50]}")
                return None
    
    async def scrape_multiple(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Scrape multiple URLs concurrently
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of dicts with url and content
        """
        # Create tasks for all URLs
        tasks = [self.scrape_url(url) for url in urls]
        
        # Run them all at once
        contents = await asyncio.gather(*tasks)
        
        # Combine URLs with their content
        results = []
        for url, content in zip(urls, contents):
            if content:  # Only include successful scrapes
                results.append({
                    "url": url,
                    "content": content
                })
        
        print(f"\n📊 Scraped {len(results)}/{len(urls)} websites successfully")
        return results