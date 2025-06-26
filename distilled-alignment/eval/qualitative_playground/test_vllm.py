#!/usr/bin/env python3
"""
Test script to verify vLLM setup with Qwen model.
"""

import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vllm_import():
    """Test if vLLM can be imported."""
    try:
        from vllm import LLM, SamplingParams
        logger.info("✓ vLLM imported successfully")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import vLLM: {e}")
        return False

def test_simple_generation():
    """Test simple text generation with vLLM."""
    try:
        from vllm import LLM, SamplingParams
        
        # Use a smaller model for testing
        logger.info("Testing with a smaller model for quick verification...")
        
        # Try with a smaller model first
        llm = LLM(
            model="microsoft/DialoGPT-small",  # Small model for testing
            trust_remote_code=True,
            dtype="float16"
        )
        
        sampling_params = SamplingParams(
            max_tokens=50,
            temperature=0.7,
            top_p=0.9
        )
        
        prompt = "Hello, how are you?"
        outputs = llm.generate([prompt], sampling_params)
        
        response = outputs[0].outputs[0].text.strip()
        logger.info(f"✓ Simple generation test passed. Response: {response}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Simple generation test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Running vLLM setup tests...")
    
    # Test 1: Import
    if not test_vllm_import():
        logger.error("vLLM import failed. Please install vLLM: pip install vllm")
        sys.exit(1)
    
    # Test 2: Simple generation
    if not test_simple_generation():
        logger.error("Simple generation test failed.")
        sys.exit(1)
    
    logger.info("✓ All tests passed! vLLM is working correctly.")
    logger.info("You can now use the playground with vLLM.")

if __name__ == "__main__":
    main() 