#!/usr/bin/env python3
"""Test Azure OpenAI configuration."""

import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI


async def test_azure_openai():
    """Test Azure OpenAI connection."""
    load_dotenv()
    
    # Get configuration from environment
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    
    if not all([api_key, endpoint, deployment]):
        print("❌ Missing Azure OpenAI configuration in .env")
        print("Required: AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME")
        return False
    
    print(f"🧪 Testing Azure OpenAI connection...")
    print(f"   Endpoint: {endpoint}")
    print(f"   Deployment: {deployment}")
    print(f"   API Version: {api_version}")
    
    try:
        # Initialize client
        client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
        
        # Test completion with GPT-5
        response = await client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a Socratic learning guide focused on exploration through metaphors and questions."},
                {"role": "user", "content": "What is consciousness? Guide me to discover this through questioning."}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        message = response.choices[0].message.content
        print(f"✅ Azure OpenAI connection successful!")
        print(f"   Response: {message}")
        
        # Test embeddings if available
        try:
            embed_response = await client.embeddings.create(
                model="text-embedding-ada-002",  # Common embedding model
                input="Test embedding"
            )
            print(f"✅ Embeddings working (dimension: {len(embed_response.data[0].embedding)})")
        except Exception as e:
            print(f"⚠️  Embeddings not available: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Azure OpenAI connection failed: {e}")
        return False


async def test_databases():
    """Test database connections."""
    print(f"\n🗄️  Testing database connections...")
    
    # Test Neo4j
    try:
        from neo4j import GraphDatabase
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "knowledge123")
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record:
                print(f"✅ Neo4j connection successful (URI: {uri})")
            driver.close()
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
    
    # Test MongoDB
    try:
        import pymongo
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        database = os.getenv("MONGODB_DATABASE", "socratic_lab")
        
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.server_info()  # Force connection
        print(f"✅ MongoDB connection successful (URI: {uri}, DB: {database})")
        client.close()
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")


async def main():
    """Main test function."""
    print("🧠 Personal Exploration Engine - Configuration Test")
    print("=" * 55)
    
    # Test Azure OpenAI
    azure_ok = await test_azure_openai()
    
    # Test databases
    await test_databases()
    
    print(f"\n📊 Test Results:")
    print(f"   Azure OpenAI: {'✅ PASS' if azure_ok else '❌ FAIL'}")
    
    if azure_ok:
        print(f"\n🎉 Configuration looks good!")
        print(f"   Next steps:")
        print(f"   1. Run: peengine init")
        print(f"   2. Start exploring: peengine start 'your topic'")
    else:
        print(f"\n⚠️  Please fix the configuration issues above.")


if __name__ == "__main__":
    asyncio.run(main())