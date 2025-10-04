# src/llm/openrouter_client.py
import httpx
import json
from typing import List, Dict, Optional
import asyncio

class OpenRouterClient:
    def __init__(self, api_key: str, model_name: str = "x-ai/grok-beta"):
        """
        Initialize OpenRouter client
        
        Args:
            api_key: OpenRouter API key
            model_name: Model to use (e.g., "x-ai/grok-beta", "anthropic/claude-3-opus", etc.)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Store conversation history for context
        self.conversation_history = []
        
        print(f"✅ OpenRouter client initialized with model: {model_name}")
        
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        search_results: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000
    ) -> str:
        """
        Generate AI response using OpenRouter
        
        Args:
            prompt: User's question
            context: Additional context (previous conversation)
            search_results: Web search results to analyze
            temperature: Creativity level (0-1)
            max_tokens: Maximum response length
            
        Returns:
            AI generated response
        """
        # Build the full prompt
        full_prompt = self._build_prompt(prompt, context, search_results)
        
        # Check prompt size
        prompt_size = len(full_prompt)
        print(f"📝 Prompt size: {prompt_size} characters")
        
        # Prepare the request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",  # Optional but recommended
            "X-Title": "AI Deep Search Engine"  # Optional
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert research assistant providing deep, comprehensive analysis based on web search results."
                },
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
                result = data['choices'][0]['message']['content']
                
                print(f"✅ Generated response ({len(result)} chars)")
                
                # Store in history
                self.conversation_history.append({
                    "user": prompt,
                    "assistant": result
                })
                
                return result
                
        except httpx.HTTPStatusError as e:
            print(f"❌ API error: {e.response.status_code} - {e.response.text}")
            
            # Handle rate limits
            if e.response.status_code == 429:
                return "Rate limit reached. Please wait a moment and try again."
            
            # Handle token limits
            if e.response.status_code == 400 and "token" in e.response.text.lower():
                print("🔄 Retrying with reduced content...")
                if search_results and len(search_results) > 3:
                    # Reduce to top 3 results
                    reduced_results = search_results[:3]
                    for r in reduced_results:
                        if 'content' in r and len(r['content']) > 1000:
                            r['content'] = r['content'][:1000] + "..."
                    return await self.generate_response(prompt, context, reduced_results, temperature, max_tokens)
            
            return f"API error: {e.response.status_code}. Please check your API key and try again."
            
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return "An unexpected error occurred. Please try again."
    
    def _build_prompt(
        self,
        user_query: str,
        context: Optional[str] = None,
        search_results: Optional[List[Dict]] = None
    ) -> str:
        """
        Build a comprehensive prompt with all available information
        """
        prompt_parts = []
        
        # Add search results if available
        if search_results:
            prompt_parts.append("## Web Search Results:\n")
            
            # Smart truncation to stay within limits
            max_content_per_source = 2000
            
            for i, result in enumerate(search_results[:10], 1):  # Max 10 sources
                prompt_parts.append(f"\n### Source {i}: {result.get('title', 'No title')}")
                
                content = result.get('content', result.get('snippet', 'No content'))
                if len(content) > max_content_per_source:
                    content = content[:max_content_per_source] + "... [truncated]"
                
                prompt_parts.append(f"Content: {content}\n")
        
        # Add conversation context if available
        if context:
            prompt_parts.append("\n## Previous Conversation:\n")
            if len(context) > 1000:
                context = context[-1000:]
            prompt_parts.append(context)
        
        # Add the user's question
        prompt_parts.append(f"\n## User Question:\n{user_query}")
        
        # Instructions for comprehensive response
        prompt_parts.append(
            "\n## Instructions:\n"
            "Based on the search results above, provide a comprehensive and detailed answer that:\n"
            "1. Directly addresses the question with depth\n"
            "2. Synthesizes information from multiple sources\n"
            "3. Includes specific examples and data\n"
            "4. Explores different perspectives\n"
            "5. Cites sources with [Source N] notation\n"
            "\nProvide a thorough, well-structured response:"
        )
        
        return "\n".join(prompt_parts)
    
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Summarize long text
        """
        prompt = f"Summarize the following text in no more than {max_length} words:\n\n{text[:5000]}\n\nSummary:"
        
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
                        "temperature": 0.5,
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
        """
        Generate related search queries for deeper research
        """
        prompt = f"""Based on the question: "{user_query}"
        
Generate {num_queries} different Google search queries to find comprehensive information.
Make them specific and diverse to cover different aspects.

Return ONLY the search queries, one per line, no numbering:"""
        
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
                        "temperature": 0.8,
                        "max_tokens": 200
                    }
                )
                response.raise_for_status()
                data = response.json()
                result = data['choices'][0]['message']['content']
                
                # Parse the response into separate queries
                queries = [q.strip() for q in result.strip().split('\n') if q.strip()]
                return queries[:num_queries]
                
        except Exception as e:
            print(f"❌ Query generation failed: {str(e)}")
            return [
                user_query,
                f"{user_query} explained",
                f"{user_query} examples"
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