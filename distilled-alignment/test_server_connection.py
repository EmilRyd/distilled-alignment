#!/usr/bin/env python3
"""
Simple test script to verify vLLM server connection and model availability.
"""

import openai
import sys

def test_connection():
    """Test connection to vLLM server."""
    try:
        # Configure OpenAI client
        openai.api_key = "EMPTY"
        openai.base_url = "http://localhost:8000/v1"
        
        client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        
        # List available models
        print("Testing connection to vLLM server...")
        models = client.models.list()
        
        print("✅ Connected successfully!")
        print(f"Available models: {[m.id for m in models.data]}")
        
        # Test a simple completion
        print("\nTesting simple completion...")
        response = client.completions.create(
            model=models.data[0].id,  # Use first available model
            prompt="Hello, how are you?",
            max_tokens=10,
            temperature=0.7
        )
        
        print(f"✅ Completion successful!")
        print(f"Response: {response.choices[0].text.strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1) 