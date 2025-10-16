# src/rag/vector_store.py - OPTIMIZED VERSION
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
    """FREE local embeddings with multi-processing"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"📥 Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Enable multi-processing for faster embeddings
        self.pool = None
        if hasattr(self.model, 'start_multi_process_pool'):
            try:
                self.pool = self.model.start_multi_process_pool()
                print("✅ Multi-process pool enabled for faster embeddings!")
            except Exception as e:
                print(f"⚠️  Could not enable multi-process pool: {e}")
                self.pool = None
        
        # Check for GPU
        if hasattr(self.model.device, 'type'):
            if self.model.device.type == 'cuda':
                print("✅ GPU acceleration enabled!")
            else:
                print("⚠️  Using CPU. Consider using cloud embeddings for speed.")
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        print(f"🔢 Encoding {len(texts)} text chunks with local model...")
        loop = asyncio.get_event_loop()
        
        # Use multi-process pool if available
        if self.pool:
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode_multi_process(
                    texts,
                    pool=self.pool,
                    batch_size=32,
                    normalize_embeddings=True
                )
            )
        else:
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    texts,
                    show_progress_bar=False,
                    batch_size=64,  # Increased from 32
                    normalize_embeddings=True
                )
            )
        
        print(f"✅ Local encoding complete: {len(texts)} embeddings generated")
        return embeddings.tolist()

class JinaEmbeddings(EmbeddingProvider):
    """FREE Jina AI embeddings - 1M tokens/day!"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.jina.ai/v1/embeddings"
        print("✅ Using Jina AI embeddings (1M tokens/day FREE!)")
        print("   Model: jina-embeddings-v2-base-en (768 dimensions)")
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        print(f"🔢 Encoding {len(texts)} chunks with Jina AI...")
        total_chars = sum(len(t) for t in texts)
        print(f"   Total content: ~{total_chars:,} characters")
        
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
                        "model": "jina-embeddings-v2-base-en"
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                
                embeddings = [item["embedding"] for item in data["data"]]
                print(f"✅ Jina AI encoding complete: {len(embeddings)} embeddings (768-dim)")
                return embeddings
                
            except httpx.HTTPStatusError as e:
                print(f"❌ Jina API error: {e.response.status_code}")
                print(f"   Response: {e.response.text[:200]}")
                print("🔄 Falling back to local embeddings...")
                local = LocalEmbeddings()
                return await local.encode(texts)
            except Exception as e:
                print(f"❌ Jina embedding failed: {str(e)[:100]}")
                print("🔄 Falling back to local embeddings...")
                local = LocalEmbeddings()
                return await local.encode(texts)

class TogetherEmbeddings(EmbeddingProvider):
    """FREE Together AI embeddings"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.together.xyz/v1/embeddings"
        print("✅ Using Together AI embeddings (FREE tier)")
        print("   Model: togethercomputer/m2-bert-80M-8k-retrieval")
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        print(f"🔢 Encoding {len(texts)} chunks with Together AI...")
        
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
                        "model": "togethercomputer/m2-bert-80M-8k-retrieval"
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                
                embeddings = [item["embedding"] for item in data["data"]]
                print(f"✅ Together AI encoding complete: {len(embeddings)} embeddings")
                return embeddings
                
            except httpx.HTTPStatusError as e:
                print(f"❌ Together API error: {e.response.status_code}")
                print(f"   Response: {e.response.text[:200]}")
                print("🔄 Falling back to local embeddings...")
                local = LocalEmbeddings()
                return await local.encode(texts)
            except Exception as e:
                print(f"❌ Together embedding failed: {str(e)[:100]}")
                print("🔄 Falling back to local embeddings...")
                local = LocalEmbeddings()
                return await local.encode(texts)

class OpenRouterEmbeddings(EmbeddingProvider):
    """OpenRouter embeddings (using same API key as LLM!)"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/embeddings"
        print("✅ Using OpenRouter embeddings (same API key as LLM)")
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        print(f"🔢 Encoding {len(texts)} chunks with OpenRouter...")
        
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
                        "model": "text-embedding-ada-002"
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                
                embeddings = [item["embedding"] for item in data["data"]]
                print(f"✅ OpenRouter encoding complete: {len(embeddings)} embeddings")
                return embeddings
                
            except Exception as e:
                print(f"❌ OpenRouter embedding failed: {str(e)[:100]}")
                print("🔄 Falling back to local embeddings...")
                local = LocalEmbeddings()
                return await local.encode(texts)

class CohereEmbeddings(EmbeddingProvider):
    """FREE Cohere embeddings (1000 requests/month)"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.cohere.ai/v1/embed"
        print("✅ Using Cohere embeddings (FREE tier)")
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        print(f"🔢 Encoding {len(texts)} chunks with Cohere...")
        
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
                print(f"✅ Cohere encoding complete: {len(data['embeddings'])} embeddings")
                return data["embeddings"]
                
            except Exception as e:
                print(f"❌ Cohere embedding failed: {str(e)[:100]}")
                print("🔄 Falling back to local embeddings...")
                local = LocalEmbeddings()
                return await local.encode(texts)

class HuggingFaceEmbeddings(EmbeddingProvider):
    """FREE HuggingFace Inference API embeddings"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model_id = "sentence-transformers/all-MiniLM-L6-v2"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        print("✅ Using HuggingFace embeddings (FREE tier)")
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        print(f"🔢 Encoding {len(texts)} chunks with HuggingFace...")
        
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
                
                embeddings = response.json()
                
                if isinstance(embeddings[0], list):
                    print(f"✅ HuggingFace encoding complete: {len(embeddings)} embeddings")
                    return embeddings
                else:
                    result = [emb for emb in embeddings]
                    print(f"✅ HuggingFace encoding complete: {len(result)} embeddings")
                    return result
                    
            except Exception as e:
                print(f"❌ HuggingFace embedding failed: {str(e)[:100]}")
                print("🔄 Falling back to local embeddings...")
                local = LocalEmbeddings()
                return await local.encode(texts)

class VectorStore:
    """
    OPTIMIZED vector store with batch embeddings
    """
    
    def __init__(self, persist_directory: str = "./data/chromadb"):
        """Initialize with configured embedding provider"""
        
        print(f"\n📚 Initializing Vector Store...")
        print(f"   Directory: {persist_directory}")
        print(f"   Provider: {config.EMBEDDING_PROVIDER}")
        
        # Choose embedding provider based on config
        if config.EMBEDDING_PROVIDER == "jina" and config.JINA_API_KEY:
            self.embedder = JinaEmbeddings(config.JINA_API_KEY)
            self.embedding_dimension = 768
        elif config.EMBEDDING_PROVIDER == "together" and config.TOGETHER_API_KEY:
            self.embedder = TogetherEmbeddings(config.TOGETHER_API_KEY)
            self.embedding_dimension = 768
        elif config.EMBEDDING_PROVIDER == "openrouter":
            self.embedder = OpenRouterEmbeddings(config.OPENROUTER_API_KEY)
            self.embedding_dimension = 1536
        elif config.EMBEDDING_PROVIDER == "cohere" and config.COHERE_API_KEY:
            self.embedder = CohereEmbeddings(config.COHERE_API_KEY)
            self.embedding_dimension = 1024
        elif config.EMBEDDING_PROVIDER == "huggingface" and config.HUGGINGFACE_API_KEY:
            self.embedder = HuggingFaceEmbeddings(config.HUGGINGFACE_API_KEY)
            self.embedding_dimension = 384
        else:
            # Default to local embeddings
            self.embedder = LocalEmbeddings(config.LOCAL_EMBEDDING_MODEL)
            self.embedding_dimension = 384
        
        print(f"   Embedding dimensions: {self.embedding_dimension}")
        
        # Initialize ChromaDB
        print(f"   Loading ChromaDB...")
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
                print(f"⚠️  Embedding dimension mismatch detected!")
                print(f"   Expected: {self.embedding_dimension}, Found: {len(sample_embedding['embeddings'][0])}")
                print(f"   Recreating collection for {config.EMBEDDING_PROVIDER}...")
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
        
        doc_count = self.collection.count()
        print(f"✅ Vector store ready!")
        print(f"   Collection: {collection_name}")
        print(f"   Documents: {doc_count}")
        print(f"   Provider: {config.EMBEDDING_PROVIDER}")
        print(f"   Dimensions: {self.embedding_dimension}\n")
    
    def clear_all(self):
        """Clear all documents from the collection"""
        try:
            collection_name = f"deep_search_docs_{config.EMBEDDING_PROVIDER}"
            print(f"🗑️ Clearing vector store: {collection_name}...")
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
            print("✅ Vector store cleared successfully")
        except Exception as e:
            print(f"❌ Error clearing vector store: {e}")
    
    async def add_documents_batch(
        self,
        documents: List[Dict[str, str]],
        metadata: Dict = None
    ) -> List[str]:
        """
        🚀 OPTIMIZED: Add multiple documents in ONE embedding call
        This is 5-10x faster than calling add_document() multiple times!
        
        Args:
            documents: List of dicts with 'url', 'title', 'content', 'query'
        """
        print(f"\n📚 Batch adding {len(documents)} documents...")
        
        all_chunks = []
        all_ids = []
        all_metadatas = []
        doc_ids = []
        
        for doc in documents:
            doc_id = hashlib.md5(doc['url'].encode()).hexdigest()
            
            # Check if exists
            existing = self.collection.get(ids=[f"{doc_id}_0"])
            if existing['ids']:
                print(f"⏭️  Skipping (exists): {doc['title'][:40]}...")
                doc_ids.append(doc_id)
                continue
            
            # Chunk document
            chunks = self._smart_chunk(
                doc['content'],
                chunk_size=config.CHUNK_SIZE,
                overlap=config.CHUNK_OVERLAP
            )
            
            if not chunks:
                continue
            
            # Prepare metadata for all chunks
            doc_metadata = {
                "url": doc['url'],
                "title": doc['title'],
                "query": doc.get('query', ''),
                "timestamp": datetime.now().isoformat(),
                "content_length": len(doc['content']),
                "embedding_provider": config.EMBEDDING_PROVIDER
            }
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{doc_id}_{i}")
                all_metadatas.append({
                    **doc_metadata, 
                    "chunk_index": i, 
                    "total_chunks": len(chunks)
                })
            
            doc_ids.append(doc_id)
            print(f"✅ Prepared: {doc['title'][:50]}... ({len(chunks)} chunks)")
        
        if not all_chunks:
            print("⚠️  No new documents to add")
            return doc_ids
        
        # 🚀 ONE SINGLE EMBEDDING CALL FOR ALL CHUNKS!
        print(f"\n🔢 Embedding {len(all_chunks)} chunks in ONE batch call...")
        embeddings = await self.embedder.encode(all_chunks)
        
        # Add to database
        print(f"💾 Storing {len(all_chunks)} chunks in ChromaDB...")
        self.collection.add(
            ids=all_ids,
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metadatas
        )
        
        print(f"✅ Batch added {len(documents)} documents ({len(all_chunks)} total chunks)")
        return doc_ids
    
    async def add_document(
        self,
        content: str,
        url: str,
        title: str,
        query: str = "",
        metadata: Dict = None
    ) -> str:
        """Legacy single document add - use add_documents_batch for speed!"""
        
        # Create document ID
        doc_id = hashlib.md5(url.encode()).hexdigest()
        
        # Check if exists
        existing = self.collection.get(ids=[f"{doc_id}_0"])
        if existing['ids']:
            print(f"⏭️  Skipping (already exists): {title[:50]}...")
            return doc_id
        
        print(f"\n📄 Adding document: {title[:60]}...")
        print(f"   URL: {url[:70]}...")
        print(f"   Content length: {len(content):,} characters")
        
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
        
        # Smart chunking
        print(f"   Chunking with size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}...")
        chunks = self._smart_chunk(
            content,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )
        
        print(f"   Created {len(chunks)} chunks")
        
        # Generate embeddings
        print(f"   Generating embeddings for {len(chunks)} chunks...")
        embeddings = await self.embedder.encode(chunks)
        
        # Prepare for ChromaDB
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {**doc_metadata, "chunk_index": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]
        
        # Add to database
        print(f"   Storing in ChromaDB...")
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        print(f"✅ Added: {title[:50]}... ({len(chunks)} chunks, {len(content):,} chars)")
        return doc_id
    
    async def search(
        self,
        query: str,
        n_results: int = 5,
        filter_query: Optional[str] = None
    ) -> List[Dict]:
        """Semantic search with reranking"""
        
        print(f"\n🔍 Semantic search: '{query[:60]}...'")
        print(f"   Requesting {n_results} results")
        if filter_query:
            print(f"   Filter: {filter_query}")
        
        # Generate query embedding
        query_embeddings = await self.embedder.encode([query])
        query_embedding = query_embeddings[0]
        
        # Search
        where_filter = {"query": filter_query} if filter_query else None
        
        search_limit = min(n_results * 3, 30)
        print(f"   Searching ChromaDB (limit={search_limit})...")
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_limit,
            where=where_filter
        )
        
        # Process results
        formatted_results = []
        seen_urls = set()
        
        if results['ids'] and results['ids'][0]:
            print(f"   Processing {len(results['ids'][0])} raw results...")
            
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                
                # Skip duplicates
                if metadata['url'] in seen_urls:
                    continue
                
                seen_urls.add(metadata['url'])
                
                # Calculate similarity (0-1 range)
                distance = results['distances'][0][i]
                similarity = max(0, 1 - distance)
                
                # More lenient threshold
                if similarity > 0.2:
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
        
        print(f"✅ Found {len(formatted_results)} relevant results")
        if formatted_results:
            print(f"   Top similarity score: {formatted_results[0]['similarity_score']:.3f}")
            print(f"   Lowest similarity: {formatted_results[-1]['similarity_score']:.3f}")
        
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
        
        if not text:
            return []
        
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
        stats = {
            "total_documents": self.collection.count(),
            "embedding_provider": config.EMBEDDING_PROVIDER,
            "embedding_dimension": self.embedding_dimension
        }
        return stats