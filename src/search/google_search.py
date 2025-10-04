import httpx
from typing import List, Dict

class GoogleSearcher:
    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    async def search(self, query: str, num_results: int = 10) -> List[Dict]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self.base_url,
                    params={
                        "key": self.api_key,
                        "cx": self.cse_id,
                        "q": query,
                        "num": min(num_results, 10)
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                
                data = response.json()
                results = []
                
                for item in data.get("items", []):
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
                
                print(f"Found {len(results)} results for '{query}'")
                return results
                
            except httpx.HTTPStatusError as e:
                print(f"Search failed: {e.response.status_code}")
                return []
            except Exception as e:
                print(f"Search error: {str(e)}")
                return []