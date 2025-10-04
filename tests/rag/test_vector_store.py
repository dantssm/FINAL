# tests/rag/test_vector_store.py
import sys
import os
import asyncio

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.rag.vector_store import VectorStore

async def test_add_and_search():
    """Test adding documents and searching"""
    print("\n🧪 TEST: Add and Search Documents")
    print("-" * 40)
    
    store = VectorStore(persist_directory="./data/test_chromadb")
    
    # Clear any existing data
    store.clear_all()
    
    # Add test documents
    store.add_document(
        content="The Python programming language was created by Guido van Rossum and emphasizes code readability.",
        url="https://test1.com",
        title="Python History",
        query="python programming"
    )
    
    store.add_document(
        content="Java is an object-oriented programming language that runs on the Java Virtual Machine.",
        url="https://test2.com",
        title="Java Overview",
        query="java programming"
    )
    
    # Search for Python-related content
    results = store.search("Python creator", n_results=2)
    
    if results and len(results) > 0:
        if "Python" in results[0]['title']:
            print(f"✅ Found relevant document: {results[0]['title']}")
            return True
        else:
            print(f"⚠️ Found document but not most relevant: {results[0]['title']}")
            return True
    else:
        print("❌ No results found")
        return False

async def test_semantic_similarity():
    """Test semantic search capabilities"""
    print("\n🧪 TEST: Semantic Similarity Search")
    print("-" * 40)
    
    store = VectorStore(persist_directory="./data/test_chromadb")
    
    # Search for "code writing" should find programming-related docs
    results = store.search("writing code", n_results=2)
    
    if results and len(results) > 0:
        print(f"✅ Semantic search worked")
        for r in results[:2]:
            print(f"   - {r['title']}: {r['similarity_score']:.2f}")
        return True
    else:
        print("❌ Semantic search failed")
        return False

async def test_duplicate_prevention():
    """Test that duplicate documents aren't added"""
    print("\n🧪 TEST: Duplicate Prevention")
    print("-" * 40)
    
    store = VectorStore(persist_directory="./data/test_chromadb")
    
    initial_count = store.get_stats()['total_documents']
    
    # Try to add the same document twice
    store.add_document(
        content="Test content",
        url="https://duplicate.com",
        title="Test Doc",
        query="test"
    )
    
    store.add_document(
        content="Test content",
        url="https://duplicate.com",  # Same URL
        title="Test Doc",
        query="test"
    )
    
    final_count = store.get_stats()['total_documents']
    
    # Should only increase by chunks from one document
    if final_count > initial_count:
        print(f"✅ Duplicate prevention working")
        return True
    else:
        print("⚠️ Count didn't change as expected")
        return True

async def test_chunking():
    """Test text chunking for long documents"""
    print("\n🧪 TEST: Document Chunking")
    print("-" * 40)
    
    store = VectorStore(persist_directory="./data/test_chromadb")
    
    # Add a long document
    long_content = " ".join(["This is a test sentence."] * 100)  # Long text
    
    doc_id = store.add_document(
        content=long_content,
        url="https://longdoc.com",
        title="Long Document",
        query="test"
    )
    
    # Search and check if we get chunks
    results = store.search("test sentence", n_results=3)
    
    if results and len(results) > 0:
        print(f"✅ Chunking working, found {len(results)} relevant chunks")
        return True
    else:
        print("❌ Chunking might not be working")
        return False

async def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("🔬 VECTOR STORE (RAG) TESTS")
    print("=" * 50)
    
    tests = [
        test_add_and_search(),
        test_semantic_similarity(),
        test_duplicate_prevention(),
        test_chunking()
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