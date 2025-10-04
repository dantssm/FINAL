# src/llm/gemini_client.py
import google.generativeai as genai
from typing import List, Dict, Optional
import asyncio

class GeminiClient:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini client
        
        Args:
            api_key: Google AI API key
            model_name: Model to use (gemini-1.5-flash is free)
        """
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={
                "temperature": 0.0,
                "top_p": 0.95,
                "max_output_tokens": 8192,
            }
        )
        
        # Store conversation history for context
        self.conversation_history = []
    
    async def generate_response(
        self,
        prompt: str,
        context: Optional[str] = None,
        search_results: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate AI response
        
        Args:
            prompt: User's question
            context: Additional context (previous conversation)
            search_results: Web search results to analyze
            
        Returns:
            AI generated response
        """
        # Build the full prompt
        full_prompt = self._build_prompt(prompt, context, search_results)
        
        try:
            # Generate response (using asyncio for consistency)
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt
            )
            
            result = response.text
            print(f"✅ Generated response ({len(result)} chars)")
            
            # Store in history
            self.conversation_history.append({
                "user": prompt,
                "assistant": result
            })
            
            return result
            
        except Exception as e:
            print(f"❌ Generation failed: {str(e)}")
            return "I'm sorry, I couldn't generate a response. Please try again."
    
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
        
        # System instruction - MORE CONCISE
        prompt_parts.append(
            "You are a helpful search assistant. Provide clear, direct answers based on search results. "
            "Be concise but comprehensive. Focus on answering the user's question directly."
        )
        
        # Add search results if available
        if search_results:
            prompt_parts.append("\n## Web Search Results:\n")
            for i, result in enumerate(search_results, 1):
                prompt_parts.append(f"\n### Source {i}: {result.get('title', 'No title')}")
                
                # Use content if available, otherwise use snippet
                content = result.get('content', result.get('snippet', 'No content'))
                # Limit content length to avoid huge prompts
                if len(content) > 800:
                    content = content[:800] + "..."
                prompt_parts.append(f"Content: {content}\n")
        
        # Add conversation context if available
        if context:
            prompt_parts.append("\n## Previous Conversation:\n")
            prompt_parts.append(context)
        
        # Add the user's question
        prompt_parts.append(f"\n## User Question:\n{user_query}")
        
        # BETTER instructions for response
        prompt_parts.append(
            "\n## Instructions:\n"
            "1. Give a DIRECT answer to the question first\n"
            "2. Use information from the search results\n"
            "3. Be concise - aim for 2-3 paragraphs max\n"
            "4. Only mention if sources don't contain info if that's truly the case\n"
            "5. Use natural language, not academic style\n"
            "6. If you found the answer, state it clearly\n"
            "7. If sources are unclear, give the best available information\n"
        )
        
        return "\n".join(prompt_parts)
    
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Summarize long text
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in words
            
        Returns:
            Summarized text
        """
        prompt = f"""Summarize the following text in no more than {max_length} words:

{text[:5000]}  # Limit input to avoid token limits

Summary:"""
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            return response.text
        except Exception as e:
            print(f"❌ Summarization failed: {str(e)}")
            return "Could not summarize the text."
    
    async def generate_search_queries(self, user_query: str, num_queries: int = 3) -> List[str]:
        """
        Generate related search queries for deeper research
        
        Args:
            user_query: Original user question
            num_queries: Number of queries to generate
            
        Returns:
            List of search queries
        """
        prompt = f"""Based on the question: "{user_query}"
        
Generate {num_queries} different Google search queries that would help find comprehensive information.
Make them specific and diverse to cover different aspects.
Think about what someone would actually type into Google to find this information.

Return ONLY the search queries, one per line, no numbering or bullets:"""
        
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt
            )
            
            # Parse the response into separate queries
            queries = [
                q.strip() 
                for q in response.text.strip().split('\n') 
                if q.strip()
            ]
            
            return queries[:num_queries]
            
        except Exception as e:
            print(f"❌ Query generation failed: {str(e)}")
            # Return variations of the original query as fallback
            return [
                user_query,
                f"{user_query} explained",
                f"{user_query} examples"
            ][:num_queries]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✅ Conversation history cleared")