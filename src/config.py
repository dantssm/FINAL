# src/config.py
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # OpenRouter API - Using MORE RELIABLE free model
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv(
        "OPENROUTER_MODEL", 
        "google/gemini-2.0-flash-exp:free"
    )
    
    # Google Search API - 100 searches/day FREE
    GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY") 
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    
    # Jina AI Reader - FREE tier (1000 requests/day)
    USE_JINA_READER = os.getenv("USE_JINA_READER", "true").lower() == "true"
    
    # Embeddings configuration
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
    
    # Local embeddings (FREE but slower)
    LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Cloud embedding API keys
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    JINA_API_KEY = os.getenv("JINA_API_KEY", "")
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
    
    # Cache settings
    CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))
    MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))
    
    # Performance settings
    MAX_CONCURRENT_SCRAPES = int(os.getenv("MAX_CONCURRENT_SCRAPES", "5"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    @classmethod
    def validate(cls):
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found in .env")
        if not cls.GOOGLE_SEARCH_API_KEY:
            raise ValueError("GOOGLE_SEARCH_API_KEY not found in .env")
        if not cls.GOOGLE_CSE_ID:
            raise ValueError("GOOGLE_CSE_ID not found in .env")
        
        # Check embedding provider configuration
        if cls.EMBEDDING_PROVIDER == "jina" and not cls.JINA_API_KEY:
            print("⚠️  WARNING: Jina selected but no API key provided. Falling back to local.")
            cls.EMBEDDING_PROVIDER = "local"
        elif cls.EMBEDDING_PROVIDER == "together" and not cls.TOGETHER_API_KEY:
            print("⚠️  WARNING: Together selected but no API key provided. Falling back to local.")
            cls.EMBEDDING_PROVIDER = "local"
        elif cls.EMBEDDING_PROVIDER == "openrouter":
            print("✅ Using OpenRouter for embeddings (same API key as LLM)")
        elif cls.EMBEDDING_PROVIDER == "cohere" and not cls.COHERE_API_KEY:
            print("⚠️  WARNING: Cohere selected but no API key provided. Falling back to local.")
            cls.EMBEDDING_PROVIDER = "local"
        elif cls.EMBEDDING_PROVIDER == "huggingface" and not cls.HUGGINGFACE_API_KEY:
            print("⚠️  WARNING: HuggingFace selected but no API key provided. Falling back to local.")
            cls.EMBEDDING_PROVIDER = "local"
            
        print("✅ Configuration loaded successfully")
        print(f"   LLM: {cls.OPENROUTER_MODEL}")
        print(f"   Embeddings: {cls.EMBEDDING_PROVIDER}")
        print(f"   Jina Reader: {'Enabled' if cls.USE_JINA_READER else 'Disabled'}")
        print(f"   Cache: {cls.MAX_CACHE_SIZE} entries, {cls.CACHE_TTL_HOURS}h TTL")
        
        # Warn about using DeepSeek
        if "deepseek" in cls.OPENROUTER_MODEL.lower():
            print("⚠️  WARNING: DeepSeek models often produce repetitive/gibberish output!")
            print("   Recommend switching to: google/gemini-2.0-flash-exp:free")

config = Config()