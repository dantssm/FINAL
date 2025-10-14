# src/rag/vector_store.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import hashlib
import json
from datetime import datetime
import asyncio

class VectorStore:
    """
    Optimized FREE vector store
    - Local embeddings (no API cost)
    - Smarter chunking (fewer chunks = faster)
    - Batch processing for speed
    """
    
    def __init__(self, persist_directory: str = "./data/chromadb"):
        """
        Initialize the vector store with FREE local embeddings
        """
        print("🚀 Loading embedding model (this may take a moment)...")
        
        # Use local embedding model - completely FREE
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Enable GPU if available for faster embeddings
        if self.embedder.device.type == 'cuda':
            print("✅ Using GPU for embeddings!")
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="deep_search_docs",
            metadata={
                "description": "Documents from deep web searches",
                "hnsw:space": "cosine"  # Use cosine similarity
            }
        )
        
        print(f"✅ Vector store ready with {self.collection.count()} documents")
    
    async def add_document(
        self,
        content: str,
        url: str,
        title: str,
        query: str = "",
        metadata: Dict = None
    ) -> str:
        """
        Add document with optimized chunking
        """
        # Create unique ID based on URL
        doc_id = hashlib.md5(url.encode()).hexdigest()
        
        # Check if document already exists
        existing = self.collection.get(ids=[doc_id + "_0"])
        if existing['ids']:
            print(f"⏭️  Document already exists: {title[:50]}...")
            return doc_id
        
        # Prepare metadata
        doc_metadata = {
            "url": url,
            "title": title,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "content_length": len(content)
        }
        if metadata:
            doc_metadata.update(metadata)
        
        # OPTIMIZATION: Smarter chunking - larger chunks = fewer embeddings = faster
        chunks = self._smart_chunk(content, chunk_size=800, overlap=100)
        
        # OPTIMIZATION: Generate embeddings in batch (async wrapper for sync operation)
        embeddings = await asyncio.to_thread(
            self.embedder.encode,
            chunks,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        embeddings = embeddings.tolist()
        
        # Prepare data for ChromaDB
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {**doc_metadata, "chunk_index": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        print(f"✅ Added: {title[:50]}... ({len(chunks)} chunks)")
        return doc_id
    
    async def search(
        self,
        query: str,
        n_results: int = 5,
        filter_query: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for relevant documents using semantic similarity
        """
        # Generate embedding for query (async wrapper)
        query_embedding = await asyncio.to_thread(
            self.embedder.encode,
            [query],
            show_progress_bar=False
        )
        query_embedding = query_embedding[0].tolist()
        
        # Search in ChromaDB
        where_filter = {"query": filter_query} if filter_query else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results * 2,  # Get more to filter duplicates
            where=where_filter
        )
        
        # Format results and remove duplicates
        formatted_results = []
        seen_urls = set()
        
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                
                # Skip duplicates
                if metadata['url'] in seen_urls:
                    continue
                
                seen_urls.add(metadata['url'])
                
                # OPTIMIZATION: Convert distance to similarity score (0-1 range)
                distance = results['distances'][0][i]
                similarity = max(0, 1 - distance)
                
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'url': metadata['url'],
                    'title': metadata['title'],
                    'similarity_score': similarity,
                    'metadata': metadata
                })
                
                # Stop when we have enough unique results
                if len(formatted_results) >= n_results:
                    break
        
        print(f"🔍 Found {len(formatted_results)} relevant docs for: {query[:50]}...")
        return formatted_results
    
    def _smart_chunk(
        self, 
        text: str, 
        chunk_size: int = 800, 
        overlap: int = 100
    ) -> List[str]:
        """
        OPTIMIZED: Larger chunks with smart splitting
        Fewer chunks = fewer embeddings = faster processing
        """
        # First try to split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph keeps us under chunk_size, add it
            if len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Save current chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # If paragraph itself is larger than chunk_size, split it
                if len(para) > chunk_size:
                    # Split by sentences
                    sentences = para.split('. ')
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 2 <= chunk_size:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + ". "
                else:
                    current_chunk = para + "\n\n"
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Fallback: if no chunks created, just split by character limit
        if not chunks:
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
        
        return chunks
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "total_documents": self.collection.count(),
            "collection_name": "deep_search_docs",
            "embedding_model": "all-MiniLM-L6-v2 (FREE)",
            "embedding_dimension": 384
        }
    
    def clear_all(self):
        """Clear all documents from the store"""
        self.client.delete_collection("deep_search_docs")
        self.collection = self.client.create_collection(
            name="deep_search_docs",
            metadata={
                "description": "Documents from deep web searches",
                "hnsw:space": "cosine"
            }
        )
        print("✅ Vector store cleared")