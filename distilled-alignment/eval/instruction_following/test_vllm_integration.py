#!/usr/bin/env python3
"""
Test script to verify vLLM integration for instruction following evaluation.
"""

import json
import tempfile
import os
from pathlib import Path

# Import the modified evaluation script
from run_vllm_eval import get_model_response
from vllm import LLM, SamplingParams


def test_vllm_integration():
    """Test the vLLM integration with a small model."""
    
    print("Testing vLLM integration...")
    
    # Use a small model for testing
    model_name = "microsoft/DialoGPT-small"
    
    try:
        # Initialize vLLM model
        print(f"Loading model {model_name}...")
        llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="float16"
        )
        print("Model loaded successfully!")
        
        # Test prompt
        test_prompt = "Hello, how are you today?"
        
        # Test the get_model_response function
        print(f"Testing with prompt: {test_prompt}")
        response = get_model_response(
            llm=llm,
            prompt=test_prompt,
            max_tokens=50,
            temperature=0.7
        )
        
        print(f"Response: {response}")
        
        if response and len(response.strip()) > 0:
            print("✓ vLLM integration test passed!")
            return True
        else:
            print("✗ vLLM integration test failed: Empty response")
            return False
            
    except Exception as e:
        print(f"✗ vLLM integration test failed: {e}")
        return False


def test_full_pipeline():
    """Test the full evaluation pipeline with a small dataset."""
    
    print("\nTesting full evaluation pipeline...")
    
    # Create temporary test data
    test_data = [
        {"prompt": "What is the capital of France?", "expected": "Paris"},
        {"prompt": "How do you make coffee?", "expected": "Instructions for making coffee"}
    ]
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = os.path.join(temp_dir, "test_input.jsonl")
        output_dir = os.path.join(temp_dir, "results")
        
        # Write test data
        with open(input_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        # Test the main function (we'll just test the model loading part)
        try:
            from run_vllm_eval import main
            import sys
            
            # Temporarily modify sys.argv for testing
            original_argv = sys.argv
            sys.argv = [
                'test_vllm_integration.py',
                '--model', 'microsoft/DialoGPT-small',
                '--input_data', input_file,
                '--output_dir', output_dir,
                '--max_tokens', '50',
                '--temperature', '0.7',
                '--trust_remote_code',
                '--debug_single_prompt'
            ]
            
            # Run the main function
            main()
            
            # Check if output files were created
            if os.path.exists(output_dir):
                print("✓ Full pipeline test passed!")
                return True
            else:
                print("✗ Full pipeline test failed: Output directory not created")
                return False
                
        except Exception as e:
            print(f"✗ Full pipeline test failed: {e}")
            return False
        finally:
            # Restore original argv
            sys.argv = original_argv


def main():
    """Run all tests."""
    print("Running vLLM integration tests...")
    
    # Test 1: Basic vLLM integration
    test1_passed = test_vllm_integration()
    
    # Test 2: Full pipeline (optional - might take longer)
    test2_passed = test_full_pipeline()
    
    if test1_passed and test2_passed:
        print("\n✓ All tests passed! vLLM integration is working correctly.")
        print("You can now use run_vllm_eval.py for instruction following evaluation.")
    else:
        print("\n✗ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 