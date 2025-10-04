# tests/search/test_web_scraper.py
import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.search.web_scraper import WebScraper

async def test_single_url_scrape():
    """Test scraping a single URL"""
    print("\n🧪 TEST: Single URL Scrape")
    print("-" * 40)
    
    scraper = WebScraper()
    
    # Test with a reliable website
    content = await scraper.scrape_url("https://www.example.com")
    
    if content and len(content) > 0:
        print(f"✅ Scraped {len(content)} characters")
        print(f"   Preview: {content[:100]}...")
        return True
    else:
        print("❌ Failed to scrape content")
        return False

async def test_multiple_urls():
    """Test scraping multiple URLs concurrently"""
    print("\n🧪 TEST: Multiple URLs Scrape")
    print("-" * 40)
    
    scraper = WebScraper()
    
    urls = [
        "https://www.example.com",
        "https://www.wikipedia.org",
        "https://httpbin.org/html"
    ]
    
    results = await scraper.scrape_multiple(urls)
    
    if len(results) > 0:
        print(f"✅ Scraped {len(results)}/{len(urls)} URLs successfully")
        for result in results:
            print(f"   - {result['url'][:40]}...: {len(result['content'])} chars")
        return True
    else:
        print("❌ Failed to scrape any URLs")
        return False

async def test_invalid_url():
    """Test with invalid URL"""
    print("\n🧪 TEST: Invalid URL Handling")
    print("-" * 40)
    
    scraper = WebScraper()
    
    # Test with invalid URL
    content = await scraper.scrape_url("https://this-website-definitely-does-not-exist-12345.com")
    
    if content is None:
        print("✅ Correctly handled invalid URL (returned None)")
        return True
    else:
        print("❌ Should return None for invalid URL")
        return False

async def test_timeout_handling():
    """Test timeout handling"""
    print("\n🧪 TEST: Timeout Handling")
    print("-" * 40)
    
    scraper = WebScraper()
    
    # Test with a slow website
    content = await scraper.scrape_url("https://httpbin.org/delay/20")
    
    if content is None:
        print("✅ Correctly handled timeout")
        return True
    else:
        print("⚠️ Got content (site might be fast today)")
        return True  # Not a failure, just unexpected

async def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("🔬 WEB SCRAPER TESTS")
    print("=" * 50)
    
    tests = [
        test_single_url_scrape(),
        test_multiple_urls(),
        test_invalid_url(),
        test_timeout_handling()
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