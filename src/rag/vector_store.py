# src/rag/vector_store.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import hashlib
import json
from datetime import datetime
import asyncio
import httpx
from src.config import config

class EmbeddingProvider:
    """Base class for embedding providers"""
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

class LocalEmbeddings(EmbeddingProvider):
    """FREE local embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"📥 Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Check for GPU
        if hasattr(self.model.device, 'type'):
            if self.model.device.type == 'cuda':
                print("✅ GPU acceleration enabled!")
            else:
                print("⚠️  Using CPU (slower). Consider using cloud embeddings.")
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(
                texts,
                show_progress_bar=False,
                batch_size=32,  # Process in batches
                normalize_embeddings=True
            )
        )
        return embeddings.tolist()

class CohereEmbeddings(EmbeddingProvider):
    """FREE Cohere embeddings (1000 requests/month)"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.cohere.ai/v1/embed"
        print("✅ Using Cohere embeddings (FREE tier)")
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "texts": texts,
                        "model": "embed-english-v3.0",
                        "input_type": "search_document",
                        "truncate": "END"
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data["embeddings"]
                
            except Exception as e:
                print(f"❌ Cohere embedding failed: {e}")
                # Fallback to local
                print("🔄 Falling back to local embeddings")
                local = LocalEmbeddings()
                return await local.encode(texts)

class VoyageEmbeddings(EmbeddingProvider):
    """FREE Voyage AI embeddings"""
    
    def __init__(self, api_key: str):
        try:
            import voyageai
            self.client = voyageai.Client(api_key=api_key)
            print("✅ Using Voyage AI embeddings")
        except ImportError:
            print("❌ Voyage AI library not installed")
            raise
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        try:
            # Use the voyageai client
            result = await asyncio.to_thread(
                self.client.embed,
                texts,
                model="voyage-lite-02-instruct"
            )
            return result.embeddings
        except Exception as e:
            print(f"❌ Voyage embedding failed: {e}")
            # Fallback to local
            print("🔄 Falling back to local embeddings")
            local = LocalEmbeddings()
            return await local.encode(texts)

class HuggingFaceEmbeddings(EmbeddingProvider):
    """FREE HuggingFace Inference API embeddings"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Using sentence-transformers model via HF Inference API
        self.model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        print("✅ Using HuggingFace embeddings (FREE tier)")
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "inputs": texts,
                        "options": {"wait_for_model": True}
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                
                # HF returns embeddings directly
                embeddings = response.json()
                
                # Ensure it's in the right format
                if isinstance(embeddings[0], list):
                    return embeddings
                else:
                    # Sometimes returns a different format
                    return [emb for emb in embeddings]
                    
            except Exception as e:
                print(f"❌ HuggingFace embedding failed: {e}")
                # Fallback to local
                print("🔄 Falling back to local embeddings")
                local = LocalEmbeddings()
                return await local.encode(texts)

class MixedbreadEmbeddings(EmbeddingProvider):
    """FREE Mixedbread AI embeddings - 25M tokens/month free!"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.mixedbread.ai/v1/embeddings"
        print("✅ Using Mixedbread AI embeddings (25M tokens/month FREE!)")
        print("   Model: mxbai-embed-large-v1 (1024 dimensions)")
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "input": texts,
                        "model": "mxbai-embed-large-v1",  # Their best model
                        "encoding_format": "float",
                        "normalized": True  # Return normalized embeddings
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract embeddings from response
                embeddings = [item["embedding"] for item in data["data"]]
                print(f"✅ Embedded {len(texts)} text chunks successfully")
                return embeddings
                
            except httpx.HTTPStatusError as e:
                print(f"❌ Mixedbread API error: {e.response.status_code}")
                print(f"   Response: {e.response.text}")
                # Fallback to local
                print("🔄 Falling back to local embeddings")
                local = LocalEmbeddings()
                return await local.encode(texts)
            except Exception as e:
                print(f"❌ Mixedbread embedding failed: {e}")
                # Fallback to local
                print("🔄 Falling back to local embeddings")
                local = LocalEmbeddings()
                return await local.encode(texts)

class VectorStore:
    """
    OPTIMIZED vector store with multiple embedding providers
    """
    
    def __init__(self, persist_directory: str = "./data/chromadb"):
        """Initialize with configured embedding provider"""
        
        # Choose embedding provider based on config
        if config.EMBEDDING_PROVIDER == "mixedbread" and config.MIXEDBREAD_API_KEY:
            self.embedder = MixedbreadEmbeddings(config.MIXEDBREAD_API_KEY)
            self.embedding_dimension = 1024  # mxbai-embed-large-v1 dimension
        elif config.EMBEDDING_PROVIDER == "huggingface" and config.HUGGINGFACE_API_KEY:
            self.embedder = HuggingFaceEmbeddings(config.HUGGINGFACE_API_KEY)
            self.embedding_dimension = 384
        elif config.EMBEDDING_PROVIDER == "cohere" and config.COHERE_API_KEY:
            self.embedder = CohereEmbeddings(config.COHERE_API_KEY)
            self.embedding_dimension = 1024
        elif config.EMBEDDING_PROVIDER == "voyage" and config.VOYAGE_API_KEY:
            self.embedder = VoyageEmbeddings(config.VOYAGE_API_KEY)
            self.embedding_dimension = 1024
        else:
            # Default to local embeddings
            self.embedder = LocalEmbeddings(config.LOCAL_EMBEDDING_MODEL)
            self.embedding_dimension = 384
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection with dimension check
        collection_name = f"deep_search_docs_{config.EMBEDDING_PROVIDER}"
        
        try:
            self.collection = self.client.get_collection(collection_name)
            # Check if dimensions match
            sample_embedding = self.collection.get(limit=1)
            if sample_embedding['embeddings'] and len(sample_embedding['embeddings'][0]) != self.embedding_dimension:
                print(f"⚠️  Embedding dimension mismatch, recreating collection for {config.EMBEDDING_PROVIDER}")
                self.client.delete_collection(collection_name)
                raise ValueError("Dimension mismatch")
        except:
            # Create new collection
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": f"Document store using {config.EMBEDDING_PROVIDER}",
                    "hnsw:space": "cosine",
                    "embedding_dimension": self.embedding_dimension,
                    "embedding_provider": config.EMBEDDING_PROVIDER
                }
            )
            print(f"📁 Created new collection: {collection_name}")
        
        print(f"✅ Vector store ready ({self.collection.count()} docs)")
        print(f"   Provider: {config.EMBEDDING_PROVIDER}")
        print(f"   Dimensions: {self.embedding_dimension}")
    
    def clear_all(self):
        """Clear all documents from the collection"""
        try:
            collection_name = f"deep_search_docs_{config.EMBEDDING_PROVIDER}"
            self.client.delete_collection(collection_name)
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": f"Document store using {config.EMBEDDING_PROVIDER}",
                    "hnsw:space": "cosine",
                    "embedding_dimension": self.embedding_dimension,
                    "embedding_provider": config.EMBEDDING_PROVIDER
                }
            )
            print("✅ Vector store cleared")
        except Exception as e:
            print(f"❌ Error clearing vector store: {e}")
    
    async def add_document(
        self,
        content: str,
        url: str,
        title: str,
        query: str = "",
        metadata: Dict = None
    ) -> str:
        """Add document with optimized chunking"""
        
        # Create document ID
        doc_id = hashlib.md5(url.encode()).hexdigest()
        
        # Check if exists
        existing = self.collection.get(ids=[f"{doc_id}_0"])
        if existing['ids']:
            return doc_id
        
        # Prepare metadata
        doc_metadata = {
            "url": url,
            "title": title,
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "content_length": len(content),
            "embedding_provider": config.EMBEDDING_PROVIDER
        }
        if metadata:
            doc_metadata.update(metadata)
        
        # Smart chunking with configured size
        chunks = self._smart_chunk(
            content,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )
        
        # Limit chunks to avoid overwhelming the system
        if len(chunks) > 10:
            chunks = chunks[:10]
            print(f"⚠️  Limited to 10 chunks for performance")
        
        # Generate embeddings
        embeddings = await self.embedder.encode(chunks)
        
        # Prepare for ChromaDB
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {**doc_metadata, "chunk_index": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]
        
        # Add to database
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        print(f"✅ Added: {title[:40]}... ({len(chunks)} chunks)")
        return doc_id
    
    async def search(
        self,
        query: str,
        n_results: int = 5,
        filter_query: Optional[str] = None
    ) -> List[Dict]:
        """Semantic search with reranking"""
        
        # Generate query embedding
        print(f"🔍 Searching for: {query[:50]}...")
        query_embeddings = await self.embedder.encode([query])
        query_embedding = query_embeddings[0]
        
        # Search
        where_filter = {"query": filter_query} if filter_query else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results * 2, 20),  # Get extra for filtering
            where=where_filter
        )
        
        # Process results
        formatted_results = []
        seen_urls = set()
        
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                
                # Skip duplicates
                if metadata['url'] in seen_urls:
                    continue
                
                seen_urls.add(metadata['url'])
                
                # Calculate similarity (0-1 range)
                distance = results['distances'][0][i]
                similarity = max(0, 1 - distance)
                
                # Only include high-quality matches
                if similarity > 0.3:  # Threshold for relevance
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'url': metadata['url'],
                        'title': metadata['title'],
                        'similarity_score': similarity,
                        'metadata': metadata
                    })
                
                if len(formatted_results) >= n_results:
                    break
        
        # Sort by similarity
        formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        print(f"🔍 Found {len(formatted_results)} relevant results")
        return formatted_results
    
    def _smart_chunk(
        self,
        text: str,
        chunk_size: int = 600,
        overlap: int = 100
    ) -> List[str]:
        """Intelligent text chunking"""
        
        # Clean text
        text = text.strip()
        
        # Try paragraph-based chunking first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # Skip very short paragraphs
            if len(para) < 50:
                continue
                
            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Split long paragraphs
                if len(para) > chunk_size:
                    sentences = para.split('. ')
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) <= chunk_size:
                            temp_chunk += sentence + ". "
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence + ". "
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = para + "\n\n"
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If no chunks created, fall back to simple splitting
        if not chunks:
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if chunk:
                    chunks.append(chunk)
        
        return chunks
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "total_documents": self.collection.count(),
            "embedding_provider": config.EMBEDDING_PROVIDER,
            "embedding_dimension": self.embedding_dimension
        }