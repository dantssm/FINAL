# src/llm/openrouter_client.py
import httpx
import json
import re
from typing import List, Dict, Optional
import asyncio

class OpenRouterClient:
    def __init__(self, api_key: str, model_name: str = "google/gemini-2.0-flash-exp:free"):
        """
        Initialize OpenRouter client - SWITCHED TO MORE RELIABLE MODEL
        
        RECOMMENDED FREE Models (in order of reliability):
        1. google/gemini-2.0-flash-exp:free (BEST - most reliable)
        2. meta-llama/llama-3.2-3b-instruct:free 
        3. qwen/qwen-2-7b-instruct:free
        4. deepseek/deepseek-chat-v3.1:free (AVOID - produces gibberish)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Better fallback chain
        self.fallback_models = [
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "qwen/qwen-2-7b-instruct:free"
        ]
        
        self.conversation_history = []
        print(f"✅ OpenRouter initialized with: {model_name}")
        
    def _validate_response(self, text: str) -> bool:
        """
        ENHANCED validation to catch bad responses
        """
        if not text or len(text.strip()) < 20:
            return False
        
        # Check for repetitive patterns (the "you know" bug)
        words = text.lower().split()
        if len(words) > 10:
            # Check if same phrase repeats more than 5 times
            for i in range(len(words) - 5):
                phrase = words[i]
                count = sum(1 for j in range(i, min(i+20, len(words))) if words[j] == phrase)
                if count > 5:
                    print(f"⚠️ Detected repetitive pattern: '{phrase}' repeated {count} times")
                    return False
        
        # Check for other repetitive patterns (2-3 word phrases)
        for window_size in [2, 3]:
            for i in range(len(words) - window_size * 3):
                phrase = ' '.join(words[i:i+window_size])
                text_segment = ' '.join(words[i:i+window_size*6])
                if text_segment.count(phrase) >= 3:
                    print(f"⚠️ Detected repetitive phrase: '{phrase}'")
                    return False
        
        # Check for gibberish indicators
        gibberish_patterns = [
            r'(\b\w+\b)(?:\s+\1){4,}',  # Same word repeated 5+ times
            r'(.{2,10})\1{4,}',  # Any pattern repeated 5+ times
            r'[a-zA-Z]{30,}',  # Very long continuous text without spaces
        ]
        
        for pattern in gibberish_patterns:
            if re.search(pattern, text.lower()):
                print(f"⚠️ Detected gibberish pattern")
                return False
        
        # Check for nonsensical endings
        nonsense_indicators = [
            "bye bye", "sweet dreams", "god bless", 
            "have a nice day", "you're welcome",
            "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
        ]
        
        text_lower = text.lower()
        # Only flag if it's JUST these phrases or very short
        if len(text) < 100:
            for indicator in nonsense_indicators:
                if indicator in text_lower:
                    print(f"⚠️ Detected nonsense indicator: {indicator}")
                    return False
        
        return True
    
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        search_results: Optional[List[Dict]] = None,
        temperature: float = 0.1,  # Lower temperature for consistency
        max_tokens: int = 2000,
        retry_count: int = 0
    ) -> str:
        """
        Generate response with better error handling and fallbacks
        """
        # Use simpler, clearer prompt
        full_prompt = self._build_simple_prompt(prompt, search_results)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "AI Deep Search"
        }
        
        # Determine which model to use based on retry count
        current_model = self.model_name
        if retry_count > 0 and retry_count <= len(self.fallback_models):
            current_model = self.fallback_models[retry_count - 1]
            print(f"🔄 Retry {retry_count}: Using {current_model}")
        
        payload = {
            "model": current_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful search assistant. Answer questions based on the provided search results. Be direct and informative."
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            "frequency_penalty": 0.5,  # Reduce repetition
            "presence_penalty": 0.5    # Encourage variety
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                data = response.json()
                result = data['choices'][0]['message']['content'].strip()
                
                # VALIDATE the response
                if not self._validate_response(result):
                    print(f"❌ Invalid response detected from {current_model}")
                    
                    # Try with next model
                    if retry_count < len(self.fallback_models):
                        return await self.generate_response(
                            prompt, context, search_results,
                            temperature=0.1, max_tokens=max_tokens,
                            retry_count=retry_count + 1
                        )
                    else:
                        return self._generate_fallback_response(prompt, search_results)
                
                print(f"✅ Valid response from {current_model} ({len(result)} chars)")
                
                # Store in history
                self.conversation_history.append({
                    "user": prompt,
                    "assistant": result
                })
                
                return result
                
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error {e.response.status_code} from {current_model}")
            
            # Try next model
            if retry_count < len(self.fallback_models):
                return await self.generate_response(
                    prompt, context, search_results,
                    temperature=0.1, max_tokens=max_tokens,
                    retry_count=retry_count + 1
                )
            
            return self._generate_fallback_response(prompt, search_results)
            
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            
            if retry_count < len(self.fallback_models):
                await asyncio.sleep(1)  # Brief delay before retry
                return await self.generate_response(
                    prompt, context, search_results,
                    temperature=0.1, max_tokens=max_tokens,
                    retry_count=retry_count + 1
                )
            
            return self._generate_fallback_response(prompt, search_results)