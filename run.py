import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 AI DEEP SEARCH ENGINE")
    print("=" * 60)
    print("\n📍 Open your browser and go to:")
    print("   http://localhost:8000")
    print("\n💡 API Documentation available at:")
    print("   http://localhost:8000/docs")
    print("\n⚡ Press Ctrl+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )