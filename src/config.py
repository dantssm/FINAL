# src/config.py
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # OpenRouter API (for LLM) - FREE tier available
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free")  # FREE model
    
    # Google Search API - 100 searches/day FREE
    GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY") 
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    
    # Jina AI Reader - FREE tier (1000 requests/day)
    USE_JINA_READER = os.getenv("USE_JINA_READER", "true").lower() == "true"
    
    # Local embeddings - FREE (no API needed)
    USE_LOCAL_EMBEDDINGS = True
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast & free local model
    
    # Simple in-memory cache (no Redis needed for free version)
    USE_MEMORY_CACHE = True
    CACHE_TTL_HOURS = 24
    
    @classmethod
    def validate(cls):
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found in .env")
        if not cls.GOOGLE_SEARCH_API_KEY:
            raise ValueError("GOOGLE_SEARCH_API_KEY not found in .env")
        if not cls.GOOGLE_CSE_ID:
            raise ValueError("GOOGLE_CSE_ID not found in .env")
        print("✅ All API keys loaded successfully")
        print(f"   Using FREE model: {cls.OPENROUTER_MODEL}")
        print(f"   Jina Reader: {'Enabled' if cls.USE_JINA_READER else 'Disabled'}")
        print(f"   Local embeddings: {cls.EMBEDDING_MODEL}")

config = Config()