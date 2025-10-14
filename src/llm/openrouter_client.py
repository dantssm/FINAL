# src/llm/openrouter_client.py
import httpx
import json
from typing import List, Dict, Optional
import asyncio

class OpenRouterClient:
    def __init__(self, api_key: str, model_name: str = "deepseek/deepseek-chat-v3.1:free"):
        """
        Initialize OpenRouter client
        
        FREE Models Available:
        - deepseek/deepseek-chat-v3.1:free (default)
        - google/gemini-2.0-flash-exp:free (recommended fallback)
        - meta-llama/llama-3.1-8b-instruct:free
        - microsoft/phi-3-mini-128k-instruct:free
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Store conversation history
        self.conversation_history = []
        
        print(f"✅ OpenRouter client initialized with model: {model_name}")
        
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        search_results: Optional[List[Dict]] = None,
        temperature: float = 0.3,  # Lower temperature for more focused responses
        max_tokens: int = 2000
    ) -> str:
        """
        Generate AI response using OpenRouter
        """
        # Build the full prompt with BETTER formatting
        full_prompt = self._build_prompt_v2(prompt, context, search_results)
        
        # Check prompt size
        print(f"📝 Prompt size: {len(full_prompt)} characters")
        
        # Prepare the request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Deep Search Engine"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                result = data['choices'][0]['message']['content'].strip()
                
                # Check if response is valid
                if len(result) < 50 or self._is_gibberish(result):
                    print("⚠️ Invalid response detected, retrying with gemini...")
                    # Try with Gemini as fallback
                    return await self._fallback_generate(prompt, context, search_results)
                
                print(f"✅ Generated response ({len(result)} chars)")
                
                # Store in history
                self.conversation_history.append({
                    "user": prompt,
                    "assistant": result
                })
                
                return result
                
        except httpx.HTTPStatusError as e:
            print(f"❌ API error: {e.response.status_code} - {e.response.text}")
            
            # Try fallback model
            if "deepseek" in self.model_name.lower():
                print("🔄 Trying fallback model: Gemini Flash...")
                return await self._fallback_generate(prompt, context, search_results)
            
            return "I encountered an error generating a response. Please try again."
            
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return "An unexpected error occurred. Please try again."
    
    def _is_gibberish(self, text: str) -> bool:
        """
        Check if response is gibberish or invalid
        """
        gibberish_indicators = [
            "you're welcome",
            "have a nice day",
            "happy new year",
            "god bless",
            "<｜begin▁of▁sentence｜>",
            "bye bye",
            "sweet dreams"
        ]
        
        text_lower = text.lower()
        
        # Check if response contains multiple gibberish indicators
        matches = sum(1 for indicator in gibberish_indicators if indicator in text_lower)
        
        return matches >= 3
    
    async def _fallback_generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        search_results: Optional[List[Dict]] = None
    ) -> str:
        """
        Fallback to Gemini Flash if primary model fails
        """
        # Save original model
        original_model = self.model_name
        
        # Try Gemini Flash (FREE and better quality)
        self.model_name = "google/gemini-2.0-flash-exp:free"
        
        full_prompt = self._build_prompt_v2(prompt, context, search_results)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Deep Search Engine"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 2000
        }
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                result = data['choices'][0]['message']['content'].strip()
                
                print(f"✅ Fallback successful with Gemini ({len(result)} chars)")
                
                # Restore original model
                self.model_name = original_model
                
                return result
                
        except Exception as e:
            print(f"❌ Fallback also failed: {str(e)}")
            self.model_name = original_model
            return "I'm having trouble generating a response right now. The search found relevant information, but I couldn't analyze it properly. Please try again or check the sources directly."
    
    def _build_prompt_v2(
        self,
        user_query: str,
        context: Optional[str] = None,
        search_results: Optional[List[Dict]] = None
    ) -> str:
        """
        IMPROVED prompt building with better structure
        """
        prompt_parts = []
        
        # Clear, direct system instruction
        prompt_parts.append(
            "You are a helpful AI assistant that provides accurate, comprehensive answers based on web search results.\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the user's question directly and thoroughly\n"
            "2. Use information from the search results provided below\n"
            "3. Write 2-4 paragraphs in clear, natural language\n"
            "4. If sources disagree, mention different perspectives\n"
            "5. Be informative and engaging\n"
            "6. Do NOT say generic pleasantries like 'have a nice day'\n"
            "7. Focus on answering the actual question\n\n"
        )
        
        # Add search results if available
        if search_results:
            prompt_parts.append("=== WEB SEARCH RESULTS ===\n")
            
            for i, result in enumerate(search_results[:8], 1):
                prompt_parts.append(f"\n--- Source {i}: {result.get('title', 'No title')} ---")
                
                content = result.get('content', result.get('snippet', 'No content'))
                
                # Limit content length but keep it substantial
                if len(content) > 1500:
                    content = content[:1500] + "..."
                
                prompt_parts.append(f"{content}\n")
        
        # Add conversation context if available
        if context:
            prompt_parts.append("\n=== PREVIOUS CONVERSATION ===\n")
            prompt_parts.append(context[:800])  # Limit context length
            prompt_parts.append("\n")
        
        # Add the user's question - make it very clear
        prompt_parts.append(f"\n=== USER QUESTION ===\n{user_query}\n")
        
        # Final instruction
        prompt_parts.append(
            "\n=== YOUR RESPONSE ===\n"
            "Based on the search results above, provide a comprehensive answer to the user's question. "
            "Write in a natural, informative style. Start answering immediately - no pleasantries.\n\n"
            "Answer:"
        )
        
        return "\n".join(prompt_parts)
    
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Summarize long text"""
        prompt = f"Summarize this text in {max_length} words or less:\n\n{text[:5000]}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 500
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data['choices'][0]['message']['content']
                
        except Exception as e:
            print(f"❌ Summarization failed: {str(e)}")
            return "Could not summarize the text."
    
    async def generate_search_queries(self, user_query: str, num_queries: int = 3) -> List[str]:
        """Generate related search queries"""
        prompt = f"""Generate {num_queries} different Google search queries to find comprehensive information about: "{user_query}"

Make them diverse and specific. Return ONLY the search queries, one per line, no numbering or explanation.

Search queries:"""
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 200
                    }
                )
                response.raise_for_status()
                data = response.json()
                result = data['choices'][0]['message']['content']
                
                # Parse the response into separate queries
                queries = [
                    q.strip() 
                    for q in result.strip().split('\n') 
                    if q.strip() and not q.strip().startswith(('#', '-', '*', '1', '2', '3'))
                ]
                
                return queries[:num_queries]
                
        except Exception as e:
            print(f"❌ Query generation failed: {str(e)}")
            return [
                user_query,
                f"{user_query} explained",
                f"{user_query} details"
            ][:num_queries]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✅ Conversation history cleared")
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        return {
            "provider": "OpenRouter",
            "model": self.model_name,
            "base_url": self.base_url
        }