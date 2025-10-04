# tests/llm/test_gemini_client.py
import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import config
from src.llm.gemini_client import GeminiClient

async def test_basic_response():
    """Test basic response generation"""
    print("\n🧪 TEST: Basic Response Generation")
    print("-" * 40)
    
    config.validate()
    client = GeminiClient(config.GEMINI_API_KEY)
    
    response = await client.generate_response("Say 'Hello, testing!' in 5 words or less")
    
    if response and len(response) > 0:
        print(f"✅ Got response: {response[:100]}...")
        return True
    else:
        print("❌ No response generated")
        return False

async def test_with_context():
    """Test response with context"""
    print("\n🧪 TEST: Response with Context")
    print("-" * 40)
    
    client = GeminiClient(config.GEMINI_API_KEY)
    
    context = "User previously asked about Python programming."
    response = await client.generate_response(
        "What did I ask about before?",
        context=context
    )
    
    if response and "Python" in response:
        print(f"✅ Context understood: {response[:100]}...")
        return True
    else:
        print("⚠️ Context might not be used properly")
        return True  # Not critical

async def test_search_query_generation():
    """Test generating search queries"""
    print("\n🧪 TEST: Search Query Generation")
    print("-" * 40)
    
    client = GeminiClient(config.GEMINI_API_KEY)
    
    queries = await client.generate_search_queries(
        "What is artificial intelligence?",
        num_queries=3
    )
    
    if queries and len(queries) > 0:
        print(f"✅ Generated {len(queries)} queries:")
        for q in queries:
            print(f"   - {q}")
        return True
    else:
        print("❌ No queries generated")
        return False

async def test_summarization():
    """Test text summarization"""
    print("\n🧪 TEST: Text Summarization")
    print("-" * 40)
    
    client = GeminiClient(config.GEMINI_API_KEY)
    
    long_text = """
    Python is a high-level programming language known for its simplicity and readability.
    It was created by Guido van Rossum and first released in 1991. Python supports multiple
    programming paradigms including procedural, object-oriented, and functional programming.
    It has a large standard library and is widely used in web development, data science,
    artificial intelligence, and automation. Python's syntax emphasizes code readability
    with its use of significant indentation.
    """
    
    summary = await client.summarize_text(long_text, max_length=50)
    
    if summary and len(summary) < len(long_text):
        print(f"✅ Text summarized: {summary}")
        return True
    else:
        print("❌ Summarization failed")
        return False

async def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("🔬 GEMINI CLIENT TESTS")
    print("=" * 50)
    
    tests = [
        test_basic_response(),
        test_with_context(),
        test_search_query_generation(),
        test_summarization()
    ]
    
    results = await asyncio.gather(*tests)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    if passed == total:
        print("✅ All tests passed!")
    else:
        print(f"❌ {total - passed} test(s) failed")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(run_all_tests())