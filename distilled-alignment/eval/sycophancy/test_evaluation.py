#!/usr/bin/env python3
"""
Test script for sycophancy evaluation functions.
This script tests the core evaluation logic without starting a vLLM server.
"""

import json
import pandas as pd
from pathlib import Path
from run_full_evaluation import load_from_jsonl, to_vllm_prompt, LANGCHAIN_EVAL_TEMPLATE


def test_data_loading():
    """Test loading data from JSONL file."""
    print("Testing data loading...")
    
    # Check if dataset exists
    dataset_path = Path(__file__).parent / "datasets/answer.jsonl"
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return False
    
    try:
        dataset = load_from_jsonl(str(dataset_path))
        print(f"✅ Successfully loaded {len(dataset)} samples")
        print(f"Sample data structure: {list(dataset[0].keys())}")
        return True
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False


def test_prompt_conversion():
    """Test prompt format conversion."""
    print("\nTesting prompt conversion...")
    
    # Sample prompt data
    sample_prompt = [
        {"type": "human", "content": "What is 2+2?"},
        {"type": "ai", "content": "Let me think about that."},
        {"type": "human", "content": "Please give me the answer."}
    ]
    
    try:
        converted = to_vllm_prompt(sample_prompt)
        print("✅ Successfully converted prompt format")
        print(f"Converted prompt preview: {converted[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Failed to convert prompt: {e}")
        return False


def test_evaluation_template():
    """Test evaluation template formatting."""
    print("\nTesting evaluation template...")
    
    try:
        formatted = LANGCHAIN_EVAL_TEMPLATE.format(
            query="What is 2+2?",
            result="The answer is 4",
            answer="4"
        )
        print("✅ Successfully formatted evaluation template")
        print(f"Template preview: {formatted[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Failed to format template: {e}")
        return False


def test_dataframe_operations():
    """Test pandas DataFrame operations used in evaluation."""
    print("\nTesting DataFrame operations...")
    
    try:
        # Simulate dataset structure
        sample_data = [
            {
                "prompt": [{"type": "human", "content": "What is 2+2?"}],
                "base": {"question": "What is 2+2?", "correct_answer": "4"}
            }
        ]
        
        # Create DataFrame
        df = pd.DataFrame(sample_data)
        df["output"] = ["The answer is 4"]
        
        # Test evaluation prompt creation
        def to_eval_prompt(row):
            query = row["base"]["question"]
            result = row["output"]
            answer = row["base"]["correct_answer"]
            return LANGCHAIN_EVAL_TEMPLATE.format(query=query, result=result, answer=answer)
        
        df["eval_prompt"] = df.apply(to_eval_prompt, axis=1)
        
        print("✅ Successfully created DataFrame with evaluation prompts")
        print(f"DataFrame shape: {df.shape}")
        return True
    except Exception as e:
        print(f"❌ Failed DataFrame operations: {e}")
        return False


def main():
    """Run all tests."""
    print("Running sycophancy evaluation tests...\n")
    
    tests = [
        test_data_loading,
        test_prompt_conversion,
        test_evaluation_template,
        test_dataframe_operations
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! The evaluation pipeline should work correctly.")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    print(f"{'='*50}")


if __name__ == "__main__":
    main() 