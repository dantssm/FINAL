# tests/search/test_google_search.py
import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import config
from src.search.google_search import GoogleSearcher

async def test_basic_search():
    """Test basic search functionality"""
    print("\n🧪 TEST: Basic Google Search")
    print("-" * 40)
    
    # Setup
    config.validate()
    searcher = GoogleSearcher(config.GOOGLE_SEARCH_API_KEY, config.GOOGLE_CSE_ID)
    
    # Test
    results = await searcher.search("OpenAI GPT", num_results=3)
    
    # Verify
    if results:
        print(f"✅ Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"  Title: {result['title'][:50]}...")
            print(f"  Link: {result['link'][:50]}...")
            print(f"  Snippet: {result['snippet'][:80]}...")
        return True
    else:
        print("❌ No results found")
        return False

async def test_empty_query():
    """Test with empty query"""
    print("\n🧪 TEST: Empty Query")
    print("-" * 40)
    
    searcher = GoogleSearcher(config.GOOGLE_SEARCH_API_KEY, config.GOOGLE_CSE_ID)
    results = await searcher.search("", num_results=1)
    
    if len(results) == 0:
        print("✅ Correctly handled empty query")
        return True
    else:
        print("❌ Should return empty results for empty query")
        return False

async def test_special_characters():
    """Test with special characters in query"""
    print("\n🧪 TEST: Special Characters")
    print("-" * 40)
    
    searcher = GoogleSearcher(config.GOOGLE_SEARCH_API_KEY, config.GOOGLE_CSE_ID)
    results = await searcher.search("Python + AI & Machine Learning", num_results=2)
    
    if results:
        print(f"✅ Handled special characters, found {len(results)} results")
        return True
    else:
        print("⚠️ No results with special characters")
        return False

async def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("🔬 GOOGLE SEARCH TESTS")
    print("=" * 50)
    
    tests = [
        test_basic_search(),
        test_empty_query(),
        test_special_characters()
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