from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
    
    GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    
    CACHE_TTL_HOURS = 24
    MAX_CACHE_SIZE = 5000
    
    MAX_CONCURRENT_SCRAPES = 10
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 50
    
    SEARCH_TIMEOUT = 15
    SCRAPE_TIMEOUT = 15
    
    @classmethod
    def validate(cls):
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found in .env")
        if not cls.GOOGLE_SEARCH_API_KEY:
            raise ValueError("GOOGLE_SEARCH_API_KEY not found in .env")
        if not cls.GOOGLE_CSE_ID:
            raise ValueError("GOOGLE_CSE_ID not found in .env")
        if not cls.JINA_API_KEY:
            raise ValueError("JINA_API_KEY not found in .env - we need this for embeddings")
        
        print("✅ Configuration loaded")
        print(f"   LLM: {cls.OPENROUTER_MODEL}")
        print(f"   Embeddings: Jina AI")
        print(f"   Cache: {cls.MAX_CACHE_SIZE} entries, {cls.CACHE_TTL_HOURS}h TTL")

config = Config()