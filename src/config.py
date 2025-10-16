# src/config.py - FIXED: Session-only cache configuration
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # OpenRouter API
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv(
        "OPENROUTER_MODEL", 
        "google/gemini-2.0-flash-exp:free"
    )
    
    # Google Search API
    GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY") 
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    
    # Jina AI Reader
    USE_JINA_READER = os.getenv("USE_JINA_READER", "true").lower() == "true"
    
    # Embeddings configuration
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
    
    # Local embeddings
    LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    
    # Cloud embedding API keys
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
    JINA_API_KEY = os.getenv("JINA_API_KEY", "")
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")
    
    # Cache settings - SESSION ONLY (no persistence)
    CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))  # 24h TTL per session
    MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "5000"))   # 5000 entries per session
    
    # Performance settings
    MAX_CONCURRENT_SCRAPES = int(os.getenv("MAX_CONCURRENT_SCRAPES", "10"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Search optimization - REMOVED hard limits to respect user parameters
    MAX_SEARCH_QUERIES = int(os.getenv("MAX_SEARCH_QUERIES", "10"))  # Increased from 4
    MAX_URLS_TO_SCRAPE = int(os.getenv("MAX_URLS_TO_SCRAPE", "30"))  # Increased from 10
    MAX_CHUNKS_FOR_LLM = int(os.getenv("MAX_CHUNKS_FOR_LLM", "30"))  # Increased from 20
    
    # Timeout settings
    SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "15"))
    SCRAPE_TIMEOUT = int(os.getenv("SCRAPE_TIMEOUT", "15"))
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
    
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
        print(f"   Cache: SESSION-ONLY (no persistence), {cls.MAX_CACHE_SIZE} entries, {cls.CACHE_TTL_HOURS}h TTL")
        print(f"   Performance: {cls.MAX_CONCURRENT_SCRAPES} concurrent scrapes, {cls.CHUNK_SIZE} chunk size")
        
        # Warn about using DeepSeek
        if "deepseek" in cls.OPENROUTER_MODEL.lower():
            print("⚠️  WARNING: DeepSeek models often produce repetitive/gibberish output!")
            print("   Recommend switching to: google/gemini-2.0-flash-exp:free")

config = Config()