from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-4-maverick:free")
    
    GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    
    JINA_API_KEY = os.getenv("JINA_API_KEY")
    
    @classmethod
    def validate(cls):
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found in .env")
        if not cls.GOOGLE_SEARCH_API_KEY:
            raise ValueError("GOOGLE_SEARCH_API_KEY not found in .env")
        if not cls.GOOGLE_CSE_ID:
            raise ValueError("GOOGLE_CSE_ID not found in .env")
        if not cls.JINA_API_KEY:
            raise ValueError("JINA_API_KEY not found in .env")
        
        print("Configuration loaded")
        print(f"LLM: {cls.OPENROUTER_MODEL}")
        print(f"Search: Google Custom Search")
        print(f"Embeddings: Jina AI")

config = Config()