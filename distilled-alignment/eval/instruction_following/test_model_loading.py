#!/usr/bin/env python3
"""
Test script to diagnose model loading issues and find optimal parameters.
"""

import subprocess
import sys
import time
import argparse

def test_model_loading(base_model, lora_adapter=None, **kwargs):
    """Test model loading with different parameters."""
    
    cmd = [
        sys.executable, "run_vllm_eval.py",
        "--base_model", base_model,
        "--input_data", "data/input_data.jsonl",  # You'll need to adjust this path
        "--output_dir", "test_results",
        "--debug_single_prompt"
    ]
    
    if lora_adapter:
        cmd.extend(["--lora_adapter", lora_adapter])
    
    # Add optimization parameters
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"Testing with command: {' '.join(cmd)}")
    print("=" * 60)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ SUCCESS - Model loaded in {end_time - start_time:.2f} seconds")
            return True
        else:
            print(f"❌ FAILED - Return code: {result.returncode}")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT - Model loading took more than 10 minutes")
        return False
    except Exception as e:
        print(f"❌ ERROR - {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test model loading with different parameters")
    parser.add_argument("--base_model", required=True, help="Base model to test")
    parser.add_argument("--lora_adapter", help="LoRA adapter to test (optional)")
    parser.add_argument("--input_data", default="data/input_data.jsonl", help="Path to input data")
    
    args = parser.parse_args()
    
    # Test configurations from most conservative to most aggressive
    configs = [
        {
            "name": "Conservative (low memory usage)",
            "params": {
                "gpu_memory_utilization": 0.7,
                "max_model_len": 2048,
                "max_num_batched_tokens": 2048,
                "max_num_seqs": 128,
                "disable_log_stats": True
            }
        },
        {
            "name": "Balanced (default settings)",
            "params": {
                "gpu_memory_utilization": 0.8,
                "max_model_len": 4096,
                "max_num_batched_tokens": 4096,
                "max_num_seqs": 256
            }
        },
        {
            "name": "Aggressive (high memory usage)",
            "params": {
                "gpu_memory_utilization": 0.9,
                "max_model_len": 8192,
                "max_num_batched_tokens": 8192,
                "max_num_seqs": 512
            }
        }
    ]
    
    print(f"Testing model loading for: {args.base_model}")
    if args.lora_adapter:
        print(f"With LoRA adapter: {args.lora_adapter}")
    print()
    
    successful_configs = []
    
    for config in configs:
        print(f"Testing: {config['name']}")
        success = test_model_loading(args.base_model, args.lora_adapter, **config['params'])
        
        if success:
            successful_configs.append(config)
        
        print()
        time.sleep(2)  # Brief pause between tests
    
    print("=" * 60)
    print("SUMMARY:")
    if successful_configs:
        print(f"✅ {len(successful_configs)} configuration(s) worked:")
        for config in successful_configs:
            print(f"  - {config['name']}")
            print(f"    Parameters: {config['params']}")
    else:
        print("❌ No configurations worked. Try:")
        print("  1. Check GPU memory with 'nvidia-smi'")
        print("  2. Try tensor parallelism with --tensor_parallel_size=2")
        print("  3. Use a smaller model or different dtype")

if __name__ == "__main__":
    main() 