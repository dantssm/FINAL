import httpx
from typing import List, Dict, Optional
import asyncio


class OpenRouterClient:
    def __init__(self, api_key: str, model_name: str = "meta-llama/llama-4-maverick:free "):
        """Set up the OpenRouter client with API key and model"""
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        self.fallback_models = [
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "qwen/qwen-2-7b-instruct:free"
        ]
        
        self.conversation_history = []
        print(f"✅ OpenRouter initialized with: {model_name}")
    
    def clear_history(self):
        self.conversation_history = []
        print("✅ Conversation history cleared")
    
    def _build_prompt(self, user_query: str, search_results: Optional[List[Dict]] = None) -> str:
        parts = []
        
        if search_results:
            parts.append("You are answering a question using web search results. Here are the sources:\n")
            
            for i, result in enumerate(search_results, 1):
                title = result.get('title', 'Untitled')
                url = result.get('url', 'No URL')
                content = result.get('content', result.get('snippet', ''))
                
                parts.append(f"\n--- Source {i} ---")
                parts.append(f"Title: {title}")
                parts.append(f"URL: {url}")
                parts.append(f"Content: {content}\n")
        
        parts.append(f"\nUser Question: {user_query}\n")
        parts.append("Instructions:")
        parts.append("- Answer the question thoroughly using the sources above")
        parts.append("- When you use information from a source, write it like: 'According to Source 1 (url), ...'")
        parts.append("- Include the URL when citing so users can verify")
        parts.append("- Provide a detailed, comprehensive answer since this is deep search")
        parts.append("- Don't just copy text, explain and synthesize the information")
        
        return "\n".join(parts)
    
    async def generate_search_queries(self, user_query: str, num_queries: int = 3) -> List[str]:
        """
        Generate smart Google search queries that approach the question from different angles
        
        The idea: one question can be answered better with multiple searches
        For example, "how does photosynthesis work?" could benefit from:
        - "photosynthesis process steps" (the how)
        - "chloroplast function photosynthesis" (the mechanism)
        - "photosynthesis light dark reaction" (the details)
        
        This gives us broader coverage than just searching the question as-is
        """
        print(f"\n🧠 Creating {num_queries} search queries for: '{user_query}'")
        
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
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,  # Higher temp for more creative queries
                "max_tokens": 200
            }
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(self.base_url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    result = data['choices'][0]['message']['content'].strip()
                    
                    queries = [line.strip() for line in result.split('\n') if line.strip()]
                    
                    if len(queries) >= num_queries:
                        print(f"✅ AI generated {len(queries)} search queries:")
                        for i, q in enumerate(queries[:num_queries], 1):
                            print(f"   {i}. {q}")
                        return queries[:num_queries]
                    else:
                        print(f"⚠️ AI only generated {len(queries)} queries, falling back to original")
                        return [user_query]
        
        except Exception as e:
            print(f"⚠️ AI query generation failed: {str(e)[:150]}")
            return [user_query]
    
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        search_results: Optional[List[Dict]] = None,
        temperature: float = 0.1,
        retry_count: int = 0
    ) -> str:
        """
        Generate a response from the AI
        Tries different models if one fails
        """
        full_prompt = self._build_prompt(prompt, search_results)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Deep Search"
        }
        
        current_model = self.model_name
        if retry_count > 0 and retry_count <= len(self.fallback_models):
            current_model = self.fallback_models[retry_count - 1]
            print(f"🔄 Retry {retry_count}: Trying {current_model}")
        
        payload = {
            "model": current_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a research assistant. Answer questions using the provided sources. Always cite your sources with URLs."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": 3000,  # Long responses for deep search
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.base_url, headers=headers, json=payload)
                response.raise_for_status()
                
                data = response.json()
                result = data['choices'][0]['message']['content'].strip()
                
                # Basic sanity check
                if len(result) < 50:
                    print(f"⚠️ Response too short from {current_model}")
                    if retry_count < len(self.fallback_models):
                        return await self.generate_response(
                            prompt, context, search_results,
                            temperature=0.1, retry_count=retry_count + 1
                        )
                    else:
                        return "I couldn't generate a good answer. The AI models might be having issues. Try again?"
                
                print(f"✅ Got response from {current_model} ({len(result)} chars)")
                
                # Save to history
                self.conversation_history.append({
                    "user": prompt,
                    "assistant": result
                })
                
                return result
                
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error {e.response.status_code} with {current_model}")
            
            if retry_count < len(self.fallback_models):
                return await self.generate_response(
                    prompt, context, search_results,
                    temperature=0.1, retry_count=retry_count + 1
                )
            
            return f"All AI models failed. Error: {e.response.status_code}"
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
            if retry_count < len(self.fallback_models):
                await asyncio.sleep(1)
                return await self.generate_response(
                    prompt, context, search_results,
                    temperature=0.1, retry_count=retry_count + 1
                )
            
            return f"Something went wrong: {str(e)}"