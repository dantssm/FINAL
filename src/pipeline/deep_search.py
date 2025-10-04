# src/pipeline/deep_search.py
"""
Deep Search Pipeline with Session-based RAG and OpenRouter
Each user session gets its own temporary knowledge base
"""

import asyncio
from typing import List, Dict, Optional
import hashlib
import os
import shutil
from datetime import datetime, timedelta
from src.config import config
from src.search.google_search import GoogleSearcher
from src.search.web_scraper import WebScraper
from src.llm.openrouter_client import OpenRouterClient
from src.rag.vector_store import VectorStore


class SessionManager:
    """Manages session-based vector stores"""
    
    def __init__(self, base_path: str = "./data/sessions"):
        self.base_path = base_path
        self.sessions = {}
        self.session_last_access = {}
        
        # Create base directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        
        # Clean old sessions on startup
        self._cleanup_old_sessions()
    
    def get_or_create_session(self, session_id: str) -> VectorStore:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            # Create session-specific directory
            session_path = os.path.join(self.base_path, session_id)
            os.makedirs(session_path, exist_ok=True)
            
            # Create session-specific vector store
            self.sessions[session_id] = VectorStore(persist_directory=session_path)
            print(f"📁 Created new session: {session_id[:8]}...")
        
        # Update last access time
        self.session_last_access[session_id] = datetime.now()
        
        return self.sessions[session_id]
    
    def delete_session(self, session_id: str):
        """Delete a specific session and its data"""
        if session_id in self.sessions:
            # Close the vector store
            del self.sessions[session_id]
            
            # Delete the session directory
            session_path = os.path.join(self.base_path, session_id)
            if os.path.exists(session_path):
                shutil.rmtree(session_path)
                print(f"🗑️ Deleted session: {session_id[:8]}...")
        
        if session_id in self.session_last_access:
            del self.session_last_access[session_id]
    
    def _cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up sessions older than max_age_hours"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Check existing session directories
        if os.path.exists(self.base_path):
            for session_dir in os.listdir(self.base_path):
                session_path = os.path.join(self.base_path, session_dir)
                
                # Check directory modification time
                if os.path.isdir(session_path):
                    mtime = datetime.fromtimestamp(os.path.getmtime(session_path))
                    if mtime < cutoff_time:
                        shutil.rmtree(session_path)
                        print(f"🧹 Cleaned old session: {session_dir[:8]}...")
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.sessions)


class DeepSearchPipeline:
    """Main Deep Search Pipeline with session-based RAG and OpenRouter"""
    
    def __init__(self):
        """Initialize pipeline with session management"""
        config.validate()
        
        # Initialize components
        self.google_searcher = GoogleSearcher(
            config.GOOGLE_SEARCH_API_KEY,
            config.GOOGLE_CSE_ID
        )
        self.web_scraper = WebScraper()
        
        # Use OpenRouter instead of Gemini
        self.llm_client = OpenRouterClient(
            config.OPENROUTER_API_KEY,
            model_name=config.OPENROUTER_MODEL
        )
        
        # Session manager for RAG stores
        self.session_manager = SessionManager()
        
        # Store search history per session
        self.session_histories = {}
        
        print(f"✅ Deep Search Pipeline initialized")
        print(f"   Using: {config.OPENROUTER_MODEL}")
    
    def generate_session_id(self) -> str:
        """Generate a new unique session ID"""
        timestamp = str(datetime.now().timestamp())
        random_str = os.urandom(8).hex()
        return hashlib.md5(f"{timestamp}{random_str}".encode()).hexdigest()[:16]
    
    async def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        depth: int = 2,
        max_results_per_search: int = 5,
        use_rag: bool = True
    ) -> Dict:
        """
        Perform deep search with session-based RAG
        
        Args:
            query: User's question
            session_id: Session identifier (optional, will generate if not provided)
            depth: How many levels of search to perform (1-3)
            max_results_per_search: Number of results per search
            use_rag: Whether to use RAG for enhanced retrieval
            
        Returns:
            Dictionary with answer, sources, and session_id
        """
        # Get or create session ID
        if not session_id:
            session_id = self.generate_session_id()
        
        # Get session-specific vector store
        vector_store = self.session_manager.get_or_create_session(session_id)
        
        print(f"\n🔍 Starting deep search")
        print(f"   Session: {session_id[:8]}...")
        print(f"   Query: '{query}'")
        print(f"   Depth: {depth}, Max results: {max_results_per_search}")
        print(f"   Session knowledge: {vector_store.get_stats()['total_documents']} docs")
        print("=" * 50)
        
        all_search_results = []
        search_queries = [query]
        rag_results = []
        
        # Step 1: Check session's RAG for existing relevant information
        if use_rag and vector_store.get_stats()['total_documents'] > 0:
            print("\n🧠 Checking session knowledge...")
            rag_results = vector_store.search(query, n_results=5)
            
            if rag_results:
                print(f"   Found {len(rag_results)} relevant documents from this session")
                
                # Add RAG results to search results
                for rag_result in rag_results:
                    all_search_results.append({
                        'title': f"💾 {rag_result['title']}",
                        'link': rag_result['url'],
                        'snippet': rag_result['content'][:200],
                        'content': rag_result['content'],
                        'from_session_rag': True,
                        'similarity_score': rag_result['similarity_score']
                    })
        
        # Step 2: Perform web searches for new information
        for iteration in range(depth):
            print(f"\n📍 Search Level {iteration + 1}/{depth}")
            print("-" * 30)
            
            if not search_queries:
                break
            
            current_query = search_queries.pop(0)
            print(f"🔎 Searching web: {current_query}")
            
            # Search Google
            search_results = await self.google_searcher.search(
                current_query,
                num_results=max_results_per_search
            )
            
            if not search_results:
                print("   No results found")
                continue
            
            # Check which URLs are already in session
            existing_urls = {r['url'] for r in rag_results} if rag_results else set()
            new_urls = []
            
            for result in search_results:
                if result['link'] not in existing_urls:
                    new_urls.append(result['link'])
                else:
                    print(f"   ⏭️ Skipping (already in session): {result['title'][:40]}...")
            
            if new_urls:
                # Scrape new content
                print(f"📄 Scraping {len(new_urls)} new websites...")
                scraped_content = await self.web_scraper.scrape_multiple(new_urls)
                
                # Process and store in session's RAG
                for search_result in search_results:
                    for scraped in scraped_content:
                        if scraped['url'] == search_result['link']:
                            search_result['content'] = scraped['content']
                            
                            # Add to session's vector store
                            if use_rag:
                                vector_store.add_document(
                                    content=scraped['content'],
                                    url=scraped['url'],
                                    title=search_result['title'],
                                    query=query
                                )
                            break
                    
                    if 'content' in search_result:
                        all_search_results.append(search_result)
            
            # Generate follow-up queries for deeper research
            if iteration < depth - 1 and all_search_results:
                print("🤔 Generating follow-up queries...")
                new_queries = await self.llm_client.generate_search_queries(
                    query,
                    num_queries=2
                )
                
                for q in new_queries:
                    if q not in search_queries and q != current_query:
                        search_queries.append(q)
                        print(f"   + Added: {q}")
        
        # Step 3: Sort results by relevance
        all_search_results.sort(
            key=lambda x: (
                -x.get('from_session_rag', False),  # Session results first
                -x.get('similarity_score', 0)  # Then by similarity
            )
        )
        
        # Step 4: Generate comprehensive answer
        print(f"\n🧠 Analyzing {len(all_search_results)} sources...")
        print(f"   - {sum(1 for r in all_search_results if r.get('from_session_rag'))} from session knowledge")
        print(f"   - {sum(1 for r in all_search_results if not r.get('from_session_rag'))} from new searches")
        
        answer = await self.llm_client.generate_response(
            prompt=query,
            search_results=all_search_results
        )
        
        # Step 5: Store in session history
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
        
        self.session_histories[session_id].append({
            "query": query,
            "answer": answer,
            "sources": len(all_search_results),
            "timestamp": datetime.now().isoformat()
        })
        
        # Prepare response
        result = {
            "query": query,
            "answer": answer,
            "session_id": session_id,
            "sources": [
                {
                    "title": r['title'],
                    "url": r['link'],
                    "snippet": r.get('snippet', '')[:200],
                    "from_session": r.get('from_session_rag', False)
                }
                for r in all_search_results
            ],
            "total_sources": len(all_search_results),
            "session_sources": sum(1 for r in all_search_results if r.get('from_session_rag')),
            "new_sources": sum(1 for r in all_search_results if not r.get('from_session_rag')),
            "search_depth": depth,
            "session_knowledge_size": vector_store.get_stats()['total_documents']
        }
        
        print(f"\n✅ Search complete!")
        print(f"   Total sources: {result['total_sources']}")
        print(f"   Session knowledge: {result['session_knowledge_size']} documents")
        
        return result
    
    async def chat(
        self,
        message: str,
        session_id: str,
        use_search: bool = True,
        use_rag: bool = True
    ) -> str:
        """
        Chat with session context
        
        Args:
            message: User's message
            session_id: Session identifier
            use_search: Whether to perform web search
            use_rag: Whether to use session's RAG
            
        Returns:
            AI response
        """
        # Build context from session history
        context = self._build_session_context(session_id)
        
        if use_search:
            # Perform search and get answer
            result = await self.search(
                message,
                session_id=session_id,
                depth=1,
                max_results_per_search=3,
                use_rag=use_rag
            )
            return result['answer']
        elif use_rag:
            # Use session's RAG without web search
            vector_store = self.session_manager.get_or_create_session(session_id)
            
            if vector_store.get_stats()['total_documents'] > 0:
                rag_results = vector_store.search(message, n_results=5)
                response = await self.llm_client.generate_response(
                    prompt=message,
                    context=context,
                    search_results=rag_results
                )
            else:
                # No documents in session yet
                response = await self.llm_client.generate_response(
                    prompt=message,
                    context=context
                )
            return response
        else:
            # Just use LLM without search or RAG
            response = await self.llm_client.generate_response(
                prompt=message,
                context=context
            )
            return response
    
    def _build_session_context(self, session_id: str) -> str:
        """Build context from session's history"""
        if session_id not in self.session_histories:
            return ""
        
        history = self.session_histories[session_id]
        if not history:
            return ""
        
        context_parts = []
        # Use last 3 interactions for context
        for item in history[-3:]:
            context_parts.append(f"Q: {item['query']}")
            context_parts.append(f"A: {item['answer'][:500]}...")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a specific session"""
        vector_store = self.session_manager.get_or_create_session(session_id)
        stats = vector_store.get_stats()
        
        stats['session_id'] = session_id
        stats['search_history_length'] = len(self.session_histories.get(session_id, []))
        stats['active_sessions'] = self.session_manager.get_active_sessions_count()
        stats['llm_model'] = self.llm_client.get_model_info()
        
        return stats
    
    def end_session(self, session_id: str):
        """End a session and clean up its data"""
        # Delete session data
        self.session_manager.delete_session(session_id)
        
        # Clear session history
        if session_id in self.session_histories:
            del self.session_histories[session_id]
        
        print(f"👋 Session {session_id[:8]}... ended and cleaned up")
    
    def clear_history(self):
        """Clear all session histories (but keep vector stores)"""
        self.session_histories = {}
        self.llm_client.clear_history()
        print("✅ All session histories cleared")