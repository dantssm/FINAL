import httpx
from typing import List, Dict, Optional
import asyncio


class OpenRouterClient:

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model_name: str = "meta-llama/llama-4-maverick:free"):
        self.api_key = api_key
        self.model_name = model_name
        
        print(f"OpenRouter initialized with: {model_name}")
    
    def _build_source_context(self, search_results: List[Dict]) -> str:
        """
        Format search results into a readable context for the LLM.
        
        Args:
            search_results: List of result dicts with title, url, content
        Returns:
            Formatted string with all sources numbered and organized
        """
        parts = ["You are answering a question using web search results. Here are the sources:\n"]
        
        for i, result in enumerate(search_results, 1):
            title = result.get('title', 'Untitled')
            url = result.get('url', 'No URL')
            content = result.get('content', result.get('snippet', ''))
            
            parts.append(f"\n--- Source {i} ---")
            parts.append(f"Title: {title}")
            parts.append(f"URL: {url}")
            parts.append(f"Content: {content}\n")
        
        return "\n".join(parts)
    
    def _build_answer_instructions(self, user_query: str) -> str:
        """
        Create the instructions that tell the AI how to generate the answer.
        
        Args:
            user_query: The user's question
        Returns:
            Formatted instructions for the AI
        """
        return f"""
User Question: {user_query}

Instructions:
- Answer the question thoroughly using the sources above
- IMPORTANT: When citing, use EXACTLY this format: "Source N" where N is the source number
Example: "According to Source 1, quantum computing uses qubits..."
Example: "This was confirmed by Source 3."
- Do NOT include URLs in your citations - just write "Source N"
- The system will automatically convert these to clickable links
- Cite sources frequently to support your claims
- Provide a detailed, comprehensive answer since this is a deep search engine
- Don't just copy text - explain, synthesize, and connect ideas from multiple sources
- If sources conflict, mention both perspectives
- Structure your answer with clear paragraphs for readability
"""
    
    async def generate_search_queries(self, user_query: str, num_queries: int = 3) -> List[str]:
        """
        Generate smart Google search queries that approach the question from different angles
        Args:
            user_query: User question
            num_queries: Number of search queries to generate
        Returns:
            List[str]: A list of search query strings
        """
        print(f"\nCreating {num_queries} search queries for: '{user_query}'")
        
        prompt = f"""You are a search query expert. Generate {num_queries} different Google search queries that will help answer this question comprehensively: "{user_query}"

Think about:
- What are the KEY concepts in this question?
- What different ANGLES could we search from?
- What SPECIFIC terms would get better results than the question itself?

IMPORTANT RULES:
- Each query should be 2-6 words
- Make them DIFFERENT from each other (cover different aspects)
- Use specific terminology, not vague words
- Think like you're actually Googling - what would find the best pages?

Just output the {num_queries} queries, one per line. No numbering, no explanations.

Example for "Why is the sky blue?":
rayleigh scattering atmosphere
sky color physics explanation
blue light wavelength scattering

Now generate queries for: "{user_query}"
"""

        try:
            headers = {"Authorization": f"Bearer {self.api_key}", 
                       "Content-Type": "application/json"}
            
            payload = {"model": self.model_name,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.7,
                       "max_tokens": 200}
            
            async with httpx.AsyncClient(timeout = 15.0) as client:

                response = await client.post(self.BASE_URL, headers = headers, json = payload)
                
                if response.status_code == 200:
                    data = response.json()
                    result = data['choices'][0]['message']['content'].strip()
                    
                    queries = [line.strip() for line in result.split('\n') if line.strip()]
                    
                    if len(queries) >= num_queries:

                        print(f"AI generated {len(queries)} search queries:")
                        for i, q in enumerate(queries[:num_queries], 1):
                            print(f"{i}. {q}")

                        return queries[:num_queries]
                    
                    else:
                        print(f"AI only generated {len(queries)} queries, falling back to original")
                        return [user_query]
        
        except Exception as e:
            print(f"AI query generation failed: {str(e)[:150]}")
            return [user_query]
    
    async def generate_response(self, 
                                prompt: str, 
                                context: Optional[str] = None, 
                                search_results: Optional[List[Dict]] = None, 
                                temperature: float = 0.1
                                ) -> str:
        """Generate a response from the AI"""
        source_context = self._build_source_context(search_results) if search_results else ""
        answer_instructions = self._build_answer_instructions(prompt)
        full_prompt = source_context + answer_instructions
        
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json",
                   "HTTP-Referer": "http://localhost:8000",
                   "X-Title": "AI Deep Search"}
        
        current_model = self.model_name
        
        payload = {
            "model": current_model,
            "messages": [{"role": "system",
                          "content": "You are a research assistant. Answer questions using the provided sources. Always cite your sources with URLs."},
                          {"role": "user",
                           "content": full_prompt}],
            "temperature": temperature,
            "max_tokens": 3000,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5
        }
        
        try:
            async with httpx.AsyncClient(timeout = 30.0) as client:

                response = await client.post(self.BASE_URL, headers=headers, json=payload)
                response.raise_for_status()
                
                data = response.json()
                result = data['choices'][0]['message']['content'].strip()
                
                if len(result) < 50:
                    print(f"Response too short from {current_model}")
                    return "I couldn't generate a comprehensive answer. The AI response was too short. Please try again."
                
                print(f"Got response from {current_model} ({len(result)} chars)")
                
                return result
                
        except httpx.HTTPStatusError as e:
            error_msg = f"OpenRouter API error: {e.response.status_code}"
            print(f"{error_msg}")
            
            try:
                error_detail = e.response.json()
                print(f"Details: {error_detail}")
            except:
                pass
            
            return f"Failed to generate answer. API returned error {e.response.status_code}. Please try again."
            
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"