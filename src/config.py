# src/config.py
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # OpenRouter API (for LLM)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "x-ai/grok-beta")  # Default to Grok
    
    # Google Search API
    GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY") 
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    
    @classmethod
    def validate(cls):
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not found in .env")
        if not cls.GOOGLE_SEARCH_API_KEY:
            raise ValueError("GOOGLE_SEARCH_API_KEY not found in .env")
        if not cls.GOOGLE_CSE_ID:
            raise ValueError("GOOGLE_CSE_ID not found in .env")
        print("✅ All API keys loaded successfully")
        print(f"   Using model: {cls.OPENROUTER_MODEL}")

config = Config()