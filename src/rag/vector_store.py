import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import hashlib
import json
from datetime import datetime

class VectorStore:
    def __init__(self, persist_directory: str = "./data/chromadb"):
        """
        Initialize the vector store with ChromaDB and embedding model
        
        Args:
            persist_directory: Where to store the vector database
        """
        # Initialize embedding model (this runs locally, no API needed)
        print("🚀 Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast, good quality
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection for storing documents
        self.collection = self.client.get_or_create_collection(
            name="deep_search_docs",
            metadata={"description": "Documents from deep web searches"}
        )
        
        print(f"✅ Vector store initialized with {self.collection.count()} documents")
    
    def add_document(
        self,
        content: str,
        url: str,
        title: str,
        query: str = "",
        metadata: Dict = None
    ) -> str:
        """
        Add a document to the vector store
        
        Args:
            content: The document text
            url: Source URL
            title: Document title
            query: The search query that led to this document
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        # Create unique ID based on URL
        doc_id = hashlib.md5(url.encode()).hexdigest()
        
        # Check if document already exists
        existing = self.collection.get(ids=[doc_id])
        if existing['ids']:
            print(f"📄 Document already exists: {title[:50]}...")
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
        
        # Split content into chunks for better retrieval
        chunks = self._split_text(content, chunk_size=500, overlap=50)
        
        # Generate embeddings for each chunk
        embeddings = self.embedder.encode(chunks).tolist()
        
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
        
        print(f"✅ Added document: {title[:50]}... ({len(chunks)} chunks)")
        return doc_id
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_query: Optional[str] = None
    ) -> List[Dict]:
        """
        Search for relevant documents using semantic similarity
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_query: Optional filter for metadata (e.g., specific source)
            
        Returns:
            List of relevant documents with scores
        """
        # Generate embedding for query
        query_embedding = self.embedder.encode([query])[0].tolist()
        
        # Search in ChromaDB
        where_filter = {"query": filter_query} if filter_query else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )
        
        # Format results
        formatted_results = []
        seen_urls = set()  # To avoid duplicate documents
        
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]
            
            # Skip if we've already seen this URL
            if metadata['url'] in seen_urls:
                continue
            
            seen_urls.add(metadata['url'])
            
            formatted_results.append({
                'content': results['documents'][0][i],
                'url': metadata['url'],
                'title': metadata['title'],
                'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                'metadata': metadata
            })
        
        print(f"🔍 Found {len(formatted_results)} relevant documents for: {query[:50]}...")
        return formatted_results
    
    def search_similar_to_document(self, doc_id: str, n_results: int = 5) -> List[Dict]:
        """
        Find documents similar to a given document
        
        Args:
            doc_id: Document ID to find similar documents for
            n_results: Number of results
            
        Returns:
            List of similar documents
        """
        # Get the document
        doc = self.collection.get(ids=[f"{doc_id}_0"])
        
        if not doc['ids']:
            return []
        
        # Use the document's embedding to search
        embedding = self.collection.get(
            ids=[f"{doc_id}_0"],
            include=['embeddings']
        )['embeddings'][0]
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results + 1  # +1 because it will include itself
        )
        
        # Format and filter out the original document
        formatted_results = []
        for i in range(len(results['ids'][0])):
            if not results['ids'][0][i].startswith(doc_id):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i]
                })
        
        return formatted_results
    
    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk in characters
            overlap: Number of characters to overlap
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('. ')
                if last_period > chunk_size * 0.5:  # Only if we're past halfway
                    end = start + last_period + 1
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "total_documents": self.collection.count(),
            "collection_name": "deep_search_docs",
            "embedding_model": "all-MiniLM-L6-v2"
        }
    
    def clear_all(self):
        """Clear all documents from the store"""
        # Delete and recreate the collection
        self.client.delete_collection("deep_search_docs")
        self.collection = self.client.create_collection(
            name="deep_search_docs",
            metadata={"description": "Documents from deep web searches"}
        )
        print("✅ Vector store cleared")