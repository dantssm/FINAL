# src/llm/openrouter_client.py - IMPROVED QUERY GENERATION
import httpx
import json
import re
from typing import List, Dict, Optional
import asyncio


def _linkify_sources_markdown(text: str, search_results: List[Dict]) -> str:
    """
    Replace ["...\" – Source N] with ["...\" – [🔗 Source N]](URL)
    using URLs from search_results (1-indexed).
    """
    url_map = {i: r.get("url") or r.get("link") or "" for i, r in enumerate(search_results[:50], 1)}
    pattern = re.compile(r'\[\s*"([^"]+)"\s*[\u2013\-]\s*Source\s+(\d+)\s*\]')

    def repl(m):
        quote = m.group(1).strip()
        n = int(m.group(2))
        url = url_map.get(n, "")
        if not url:
            return m.group(0)
        return f'["{quote}" – [🔗 Source {n}]]({url})'

    return pattern.sub(repl, text)


def _linkify_sources_html(text: str, search_results: List[Dict]) -> str:
    """
    Detect both ["quote" – Source N] and (Source Nhttps://...) patterns
    and turn them into clickable HTML links (<a class="src-btn">Source N</a>).
    """
    import re

    # ✅ Мапа: Source N → URL
    url_map = {i: r.get("url") or r.get("link") or "" for i, r in enumerate(search_results[:50], 1)}

    # ✅ Патерни:
    patterns = [
        # 1. ["..." – Source 3]
        re.compile(r'\[\s*"([^"]+)"\s*[\u2013\-]\s*Source\s+(\d+)\s*\]'),
        # 2. Source 3(https://example.com)
        re.compile(r'Source\s+(\d+)\s*\((https?://[^\s)]+)\)')
    ]

    # ✅ Замінюємо ["..." – Source N]
    def replace_bracket_style(m):
        quote = m.group(1).strip()
        n = int(m.group(2))
        url = url_map.get(n, "")
        if not url:
            return m.group(0)
        return f'["{quote}" – <a class="src-btn" href="{url}" target="_blank" rel="noopener">Source {n}</a>]'

    # ✅ Замінюємо Source N(https://...)
    def replace_inline_url(m):
        n = int(m.group(1))
        url = url_map.get(n, m.group(2))
        return f'<a class="src-btn" href="{url}" target="_blank" rel="noopener">Source {n}</a>'

    # 1. ["..." – Source N]
    text = patterns[0].sub(replace_bracket_style, text)
    # 2. Source N(https://...)
    text = patterns[1].sub(replace_inline_url, text)

    return text


class OpenRouterClient:
    def __init__(self, api_key: str, model_name: str = "google/gemini-2.0-flash-exp:free"):
        """
        Initialize OpenRouter client
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
    
    def get_model_info(self):
        """Get current model info"""
        return {
            "provider": "OpenRouter",
            "model": self.model_name,
            "fallback_models": self.fallback_models
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✅ Conversation history cleared")
        
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
    
    def _build_simple_prompt(self, user_query: str, search_results: Optional[List[Dict]] = None) -> str:
        """Build a simple, clear prompt"""
        prompt_parts = []
        
        # Add search results if available
        if search_results:
            prompt_parts.append("Based on the following search results:\n")
            for i, result in enumerate(search_results, 1):
                prompt_parts.append(f"\n### Source {i}: {result.get('title', 'No title')}")
                
                # Use content if available, otherwise use snippet
                content = result.get('content', result.get('snippet', 'No content'))
                # Limit content length to avoid huge prompts
                if len(content) > 800:
                    content = content[:800] + "..."
                prompt_parts.append(f"Content: {content}\n")
        
        # Add the user's question
        prompt_parts.append(f"\nAnswer this question: {user_query}")
        prompt_parts.append("\nProvide a clear, direct, and informative answer based on the search results above.")
        
        return "\n".join(prompt_parts)
    
    def _generate_fallback_response(self, prompt: str, search_results: Optional[List[Dict]] = None) -> str:
        """Generate a fallback response when all models fail"""
        if search_results and len(search_results) > 0:
            # Try to extract key information from search results
            response = f"Based on the search results for '{prompt}':\n\n"
            
            for i, result in enumerate(search_results[:3], 1):
                title = result.get('title', '')
                snippet = result.get('snippet', result.get('content', ''))[:200]
                if title or snippet:
                    response += f"{i}. {title}\n"
                    if snippet:
                        response += f"   {snippet}...\n\n"
            
            return response.strip()
        else:
            return f"I apologize, but I'm having trouble generating a response for '{prompt}'. Please try again or rephrase your question."
    
    async def generate_search_queries(self, user_query: str, num_queries: int = 3) -> List[str]:
        """
        IMPROVED: Generate better search queries using the LLM
        Falls back to smart variations if LLM fails
        """
        print(f"\n🧠 Generating {num_queries} search queries for: '{user_query}'")
        
        # Try to use LLM for better query generation
        prompt = f"""Generate {num_queries} diverse Google search queries to find comprehensive information about: "{user_query}"

Requirements:
- Make queries SHORT and PRECISE (2-5 words)
- Make each query UNIQUE and cover different aspects
- Use proper grammar
- NO explanations, just the queries
- Return ONLY the queries, one per line

Examples:
For "how does quantum computing work?":
quantum computing basics
quantum computer mechanics
quantum vs classical computing

Now generate {num_queries} queries for: "{user_query}"

Queries:"""

        try:
            # Try to get LLM-generated queries
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 150
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    result = data['choices'][0]['message']['content'].strip()
                    
                    # Parse queries
                    queries = [
                        q.strip() 
                        for q in result.split('\n') 
                        if q.strip() and not q.strip().startswith(('#', '-', '*', '•'))
                    ]
                    
                    # Clean up queries (remove numbering)
                    cleaned_queries = []
                    for q in queries:
                        # Remove leading numbers and dots
                        q = re.sub(r'^\d+[\.\)]\s*', '', q)
                        q = q.strip('"\'')
                        if q and len(q) > 3:
                            cleaned_queries.append(q)
                    
                    if len(cleaned_queries) >= num_queries:
                        print(f"✅ LLM generated {len(cleaned_queries)} queries successfully")
                        for i, q in enumerate(cleaned_queries[:num_queries], 1):
                            print(f"   {i}. {q}")
                        return cleaned_queries[:num_queries]
        
        except Exception as e:
            print(f"⚠️ LLM query generation failed: {str(e)[:100]}")
        
        # FALLBACK: Smart query variations
        print("🔄 Using smart fallback query generation...")
        
        # Parse the user query to create better variations
        query_lower = user_query.lower()
        
        variations = [user_query]  # Start with original
        
        # Extract key terms (nouns, important words)
        words = user_query.split()
        key_words = [w for w in words if len(w) > 3 and w.lower() not in 
                     ['what', 'who', 'when', 'where', 'why', 'how', 'does', 'is', 'are', 'the']]
        
        if key_words:
            key_phrase = ' '.join(key_words)
            
            # Add smart variations
            if 'what' in query_lower or 'who' in query_lower:
                variations.append(f"{key_phrase} definition")
                variations.append(f"{key_phrase} explained")
            
            if 'how' in query_lower:
                variations.append(f"{key_phrase} tutorial")
                variations.append(f"{key_phrase} guide")
            
            if 'why' in query_lower:
                variations.append(f"{key_phrase} reasons")
                variations.append(f"{key_phrase} explanation")
            
            # Add comparison and example queries
            variations.append(f"{key_phrase} examples")
            variations.append(f"best {key_phrase}")
            variations.append(f"{key_phrase} overview")
        
        # Ensure uniqueness
        unique_variations = []
        seen = set()
        for v in variations:
            v_clean = v.lower().strip()
            if v_clean not in seen:
                seen.add(v_clean)
                unique_variations.append(v)
        
        result = unique_variations[:num_queries]
        
        print(f"✅ Generated {len(result)} fallback queries:")
        for i, q in enumerate(result, 1):
            print(f"   {i}. {q}")
        
        return result
    
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        search_results: Optional[List[Dict]] = None,
        temperature: float = 0.1,
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
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5
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

                if search_results:
                    result = _linkify_sources_html(result, search_results)
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
                await asyncio.sleep(1)
                return await self.generate_response(
                    prompt, context, search_results,
                    temperature=0.1, max_tokens=max_tokens,
                    retry_count=retry_count + 1
                )
            
            return self._generate_fallback_response(prompt, search_results)
    
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