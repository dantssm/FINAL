import chromadb
from chromadb.config import Settings
from typing import List, Dict
import hashlib
import httpx

from src.config import config


class VectorStore:
    """
    Vector database for storing and searching document chunks
    
    How it works:
    1. Takes documents (scraped web pages)
    2. Splits them into smaller chunks (easier to search)
    3. Converts each chunk to a vector (list of numbers) using Jina embeddings
    4. Stores vectors in ChromaDB
    5. When you search, converts your query to a vector and finds similar ones
    """
    
    def __init__(self, persist_directory: str = "./data/chromadb"):
        print(f"\n📚 Initializing Vector Store...")
        print(f"   Directory: {persist_directory}")
        print(f"   Using Jina AI embeddings")
        
        self.jina_api_key = config.JINA_API_KEY
        self.jina_url = "https://api.jina.ai/v1/embeddings"
        self.embedding_dimension = 768
        
        print(f"   Loading ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        collection_name = "deep_search_docs"
        
        try:
            self.collection = self.client.get_collection(collection_name)
            print(f"   Loaded existing collection")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": "Documents for deep search",
                    "hnsw:space": "cosine"
                }
            )
            print(f"   Created new collection")
        
        doc_count = self.collection.count()
        print(f"✅ Vector store ready with {doc_count} documents!\n")
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        print(f"🔢 Getting embeddings for {len(texts)} chunks from Jina...")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.jina_url,
                headers={
                    "Authorization": f"Bearer {self.jina_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "input": texts,
                    "model": "jina-embeddings-v2-base-en"
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise Exception(f"Jina API error: {response.status_code}")
            
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            
            print(f"✅ Got {len(embeddings)} embeddings from Jina")
            return embeddings
    
    async def add_documents_batch(self, documents: List[Dict[str, str]]) -> List[str]:
        print(f"\n📚 Adding {len(documents)} documents...")
        
        all_chunks = []      # The actual text chunks
        all_ids = []         # Unique IDs for each chunk
        all_metadatas = []   # Extra info about each chunk
        doc_ids = []         # IDs of the documents we process
        
        for doc in documents:
            doc_id = hashlib.md5(doc['url'].encode()).hexdigest()
            
            existing = self.collection.get(ids=[f"{doc_id}_0"])
            if existing['ids']:
                print(f"⏭️  Skipping (already have it): {doc['title'][:50]}...")
                doc_ids.append(doc_id)
                continue
            
            chunks = self.chunk_text(doc['content'])
            
            if not chunks:
                continue
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{doc_id}_{i}")  # doc_id_0, doc_id_1, doc_id_2...
                all_metadatas.append({
                    "url": doc['url'],
                    "title": doc['title'],
                    "query": doc.get('query', ''),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
            
            doc_ids.append(doc_id)
            print(f"✅ Prepared: {doc['title'][:60]}... ({len(chunks)} chunks)")
        
        if not all_chunks:
            print("⚠️  No new documents to add")
            return doc_ids
        
        print(f"\n🔢 Converting {len(all_chunks)} chunks to vectors...")
        embeddings = await self.get_embeddings(all_chunks)
        
        print(f"💾 Storing in database...")
        self.collection.add(
            ids=all_ids,
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metadatas
        )
        
        print(f"✅ Added {len(documents)} documents ({len(all_chunks)} chunks total)")
        return doc_ids
    
    async def search(self, query: str, n_results: int = 5) -> List[Dict]:
        print(f"\n🔍 Searching for: '{query[:60]}...'")
        
        query_embedding = await self.get_embeddings([query])

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results * 2
        )
        
        formatted = []
        seen_urls = set()
        
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                
                if metadata['url'] in seen_urls:
                    continue
                seen_urls.add(metadata['url'])
                
                distance = results['distances'][0][i]
                similarity = max(0, 1 - distance)
                
                formatted.append({
                    'content': results['documents'][0][i],
                    'url': metadata['url'],
                    'title': metadata['title'],
                    'similarity_score': similarity
                })
                
                if len(formatted) >= n_results:
                    break
        
        print(f"✅ Found {len(formatted)} relevant chunks")
        if formatted:
            print(f"   Best match: {formatted[0]['similarity_score']:.3f}")
        
        return formatted
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 50) -> List[str]:
        text = text.strip()
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk:
                chunks.append(chunk)
            
            start += chunk_size - overlap
        
        return chunks
    
    def get_stats(self) -> Dict:
        """Get info about what's in the vector store"""
        return {
            "total_documents": self.collection.count(),
            "embedding_provider": "jina",
            "embedding_dimension": self.embedding_dimension
        }