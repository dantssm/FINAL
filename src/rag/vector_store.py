import chromadb
from typing import List, Dict
import hashlib
import httpx

from src.config import config


class VectorStore:

    COLLECTION_NAME = "deep_search_docs"
    JINA_API_URL = "https://api.jina.ai/v1/embeddings"
    JINA_MODEL = "jina-embeddings-v2-base-en"
    
    def __init__(self, persist_directory: str = "./data/chromadb"):

        print(f"\nInitializing Vector Store...")
        print(f"Directory: {persist_directory}")
        print(f"Using Jina AI embeddings")
        
        self.jina_api_key = config.JINA_API_KEY
        self.embedding_dimension = 768
        
        print(f"    Loading ChromaDB...")
        self.client = chromadb.Client()

        self.collection = self.client.create_collection(name=self.COLLECTION_NAME, 
                                                        metadata={"description": "Documents for deep search", 
                                                                    "hnsw:space": "cosine"})
        
        print(f"Vector store is ready!\n")
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get cloud embeddings from Jina AI

        Args:
            texts (List[str]): List of texts to embed
        Returns:
            List[List[float]]: List of embeddings
        """
        print(f"Getting embeddings for {len(texts)} chunks from Jina...")
        
        async with httpx.AsyncClient() as client:

            response = await client.post(self.JINA_API_URL, 
                                         headers={"Authorization": f"Bearer {self.jina_api_key}", 
                                                  "Content-Type": "application/json"}, 
                                         json={"input": texts, 
                                               "model": self.JINA_MODEL}, 
                                         timeout=30.0)
            
            if response.status_code != 200:
                raise Exception(f"Jina API error: {response.status_code}")
            
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            
            print(f"Reveived {len(embeddings)} embeddings from Jina")
            return embeddings
    
    async def add_documents_batch(self, documents: List[Dict[str, str]]):
        """
        Add a batch of documents to the vector store

        Args:
            documents (List[Dict[str, str]]): List of documents
                [{"url": str, "title": str, "content": str}, ...]
        """
        print(f"\n📚 Adding {len(documents)} documents...")
        
        all_chunks = []
        all_metadatas = []
        all_ids = []
        
        for doc in documents:

            chunks = self.chunk_text(doc['content'])
            if not chunks:
                continue
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({"url": doc['url'],
                                      "title": doc['title'],
                                      "query": doc.get('query', ''),
                                      "chunk_index": i,
                                      "total_chunks": len(chunks)})
            
                chunk_id = hashlib.md5(f"{doc['url']}_{i}".encode()).hexdigest()
                all_ids.append(chunk_id)

            print(f"Prepared: {doc['title'][:60]}... ({len(chunks)} chunks)")
        
        if not all_chunks:
            print("No new documents to add")
            return
        
        print(f"\nConverting {len(all_chunks)} chunks to vectors...")
        embeddings = await self.get_embeddings(all_chunks)
        
        print(f"Storing in database...")
        self.collection.add(ids = all_ids,
                            embeddings = embeddings,
                            documents = all_chunks,
                            metadatas = all_metadatas)
        
        print(f"Added {len(documents)} documents ({len(all_chunks)} chunks total)")
    
    async def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """
        Search relevant documents in the vector store [needs fixing the logic in future]
        [needs fixing the logic in future]

        Args:
            query (str): Search query
            n_results (int): Number of results to return
        Returns:
            List[Dict]: List of search results
                [{"content": str, "url": str, "title": str, "similarity_score": float}, ...]
        """
        print(f"\nSearching for: '{query[:60]}...'")
        
        query_embedding = await self.get_embeddings([query])

        results = self.collection.query(query_embeddings = query_embedding, 
                                        n_results = n_results * 2)
        
        """
        results = {'ids': [['id1', 'id2', ...]],
                   'distances': [[0.1, 0.2, ...]],
                   'documents': [['text1', 'text2', ...]],
                   'metadatas': [[{...}, {...}, ...]]}
        """

        search_results = []
        seen_urls = set()
        
        if results['ids'] and results['ids'][0]:

            for i in range(len(results['ids'][0])):

                metadata = results['metadatas'][0][i]
                
                if metadata['url'] in seen_urls:
                    continue
                seen_urls.add(metadata['url'])
                
                distance = results['distances'][0][i]
                similarity = max(0, 1 - distance)
                
                search_results.append({'content': results['documents'][0][i],
                                  'url': metadata['url'],
                                  'title': metadata['title'],
                                  'similarity_score': similarity})
                
                if len(search_results) >= n_results:
                    break
        
        print(f"Found {len(search_results)} relevant chunks")
        
        return search_results
    
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 50) -> List[str]:
        """
        Split text into smaller pieces
        [maybe use langchain text splitter in future]

        Args:
            text (str): Text to split into chunks
            chunk_size (int): Size of the chunk
            overlap (int): Number of overlapping characters between chunks
        Returns:
            List[str]: List of chunks
        """
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
        """Get info about what's in the vector store for monitoring"""
        return {
            "total_documents": self.collection.count(),
            "embedding_provider": "jina",
            "embedding_dimension": self.embedding_dimension
        }
    