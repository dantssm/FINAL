# tests/pipeline/test_deep_search.py
import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.pipeline.deep_search import DeepSearchPipeline

async def test_basic_search():
    """Test basic search functionality"""
    print("\n🧪 TEST: Basic Search (Depth 1)")
    print("-" * 40)
    
    pipeline = DeepSearchPipeline()
    
    result = await pipeline.search(
        "What is REST API?",
        depth=1,
        max_results_per_search=2
    )
    
    if result and 'answer' in result and len(result['answer']) > 0:
        print(f"✅ Got answer ({len(result['answer'])} chars)")
        print(f"   Sources: {result['total_sources']}")
        print(f"   Preview: {result['answer'][:150]}...")
        return True
    else:
        print("❌ No answer generated")
        return False

async def test_deep_search():
    """Test multi-level deep search"""
    print("\n🧪 TEST: Deep Search (Depth 2)")
    print("-" * 40)
    
    pipeline = DeepSearchPipeline()
    
    result = await pipeline.search(
        "How to learn programming?",
        depth=2,
        max_results_per_search=2
    )
    
    if result and result['total_sources'] > 0:
        print(f"✅ Deep search completed")
        print(f"   Total sources: {result['total_sources']}")
        print(f"   Search depth: {result['search_depth']}")
        return True
    else:
        print("❌ Deep search failed")
        return False

async def test_chat_with_context():
    """Test chat functionality with context"""
    print("\n🧪 TEST: Chat with Context")
    print("-" * 40)
    
    pipeline = DeepSearchPipeline()
    
    # First, do a search to build context
    await pipeline.search("What is Python?", depth=1, max_results_per_search=2)
    
    # Now chat with context
    response = await pipeline.chat(
        "Tell me more about what we just discussed",
        use_search=False
    )
    
    if response and len(response) > 0:
        print(f"✅ Chat response with context")
        print(f"   Response: {response[:150]}...")
        return True
    else:
        print("❌ Chat failed")
        return False

async def test_chat_with_search():
    """Test chat with automatic search"""
    print("\n🧪 TEST: Chat with Search")
    print("-" * 40)
    
    pipeline = DeepSearchPipeline()
    
    response = await pipeline.chat(
        "What are the latest AI trends?",
        use_search=True
    )
    
    if response and len(response) > 0:
        print(f"✅ Chat with search successful")
        print(f"   Response: {response[:150]}...")
        return True
    else:
        print("❌ Chat with search failed")
        return False

async def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("🔬 DEEP SEARCH PIPELINE TESTS")
    print("=" * 50)
    
    tests = [
        test_basic_search(),
        test_deep_search(),
        test_chat_with_context(),
        test_chat_with_search()
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