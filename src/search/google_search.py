import httpx
import asyncio
from typing import List, Dict

class GoogleSearcher:

    def __init__(self, api_key: str, cse_id: str):

        self.api_key = api_key
        self.cse_id = cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    async def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """"
        Performs a Google search for the given query.

        Args:
            query (str): The search query.
            num_results (int): The number of search results (max = 100, default = 10).

        Returns:
            List[Dict]: A list of dictionaries containing search results.
            [result1: {"title": str, "link": str, "snippet": str}, ...]
        """

        async with httpx.AsyncClient() as client:

            results = []
            start_index = 1
            num_to_fetch = min(num_results, 100)

            while len(results) < num_to_fetch:

                try:

                    response = await client.get(
                        self.base_url, 
                        params={"key": self.api_key, 
                                "cx": self.cse_id, "q": query, 
                                "num": 10, 
                                "start": start_index}, 
                        timeout=10.0)
                    
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    for item in data.get("items", []):

                        results.append({"title": item.get("title", ""), 
                                        "link": item.get("link", ""), 
                                        "snippet": item.get("snippet", "")})
                        
                        if len(results) >= num_to_fetch:
                            break
                    
                    start_index += 10

                    await asyncio.sleep(0.1)
                    
                    print(f"Found {len(results)} results for '{query}'")
                    return results
                
                except httpx.HTTPStatusError as e:
                    print(f"Search failed: {e.response.status_code}")
                    return []
            
                except Exception as e:
                    print(f"Search error: {str(e)}")
                    return []