#!/usr/bin/env python3
"""
Complete pipeline for running instruction following evaluation.

This script:
1. Takes a HuggingFace model ID (base model or LoRA)
2. Starts a vLLM server with the model
3. Runs the evaluation using the client-server approach
4. Cleans up the server

Usage:
    python run_eval_pipeline.py <model_id> [--input_data path] [--output_dir path]
    
Examples:
    # Base model
    python run_eval_pipeline.py "meta-llama/Llama-3.1-8B"
    
    # LoRA adapter
    python run_eval_pipeline.py "EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-ec56b28c"
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import requests
from pathlib import Path
from typing import Optional, Tuple

# Add the eval directory to the path
sys.path.append(str(Path(__file__).parent / "eval" / "instruction_following"))


def activate_venv():
    """Activate the virtual environment."""
    venv_path = Path(__file__).parent / ".venv"
    if venv_path.exists():
        # Update PATH to include venv
        venv_bin = venv_path / "bin"
        if venv_bin.exists():
            os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"
            os.environ["VIRTUAL_ENV"] = str(venv_path)
            print(f"Activated virtual environment: {venv_path}")
        else:
            print(f"Warning: Virtual environment bin directory not found at {venv_bin}")
    else:
        print(f"Warning: Virtual environment not found at {venv_path}")


def parse_model_id(model_id: str) -> Tuple[str, Optional[str]]:
    """
    Parse model ID to determine if it's a base model or LoRA adapter.
    
    Returns:
        Tuple of (base_model, lora_adapter) where lora_adapter is None for base models
    """
    # For now, assume any model ID is a LoRA adapter if it contains certain patterns
    # This is a simple heuristic - you might want to make this more sophisticated
    
    # Check if it looks like a LoRA adapter (contains specific patterns)
    lora_indicators = ["lora", "adapter", "finetuned", "reference"]
    is_lora = any(indicator in model_id.lower() for indicator in lora_indicators)
    
    if is_lora:
        # For LoRA adapters, we need to determine the base model
        # This is a simplified approach - you might need to adjust based on your specific adapters
        if "llama-3.1-8b" in model_id.lower():
            base_model = "meta-llama/Llama-3.1-8B"
        elif "llama-2-7b" in model_id.lower():
            base_model = "meta-llama/Llama-2-7b-chat-hf"
        elif "llama-2-13b" in model_id.lower():
            base_model = "meta-llama/Llama-2-13b-chat-hf"
        elif "llama-2-70b" in model_id.lower():
            base_model = "meta-llama/Llama-2-70b-chat-hf"
        else:
            # Default to Llama-3.1-8B if we can't determine
            base_model = "meta-llama/Llama-3.1-8B"
            print(f"Warning: Could not determine base model for {model_id}, using {base_model}")
        
        return base_model, model_id
    else:
        # Assume it's a base model
        return model_id, None


def start_server(base_model: str, lora_adapter: Optional[str] = None) -> subprocess.Popen:
    """Start the vLLM server."""
    script_path = Path(__file__).parent / "scripts" / "serve_model.sh"
    
    cmd = [str(script_path), base_model]
    if lora_adapter:
        cmd.append(lora_adapter)
    
    print(f"Starting vLLM server with command: {' '.join(cmd)}")
    
    # Start the server process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    return process


def wait_for_server(server_url: str = "http://localhost:8000", timeout: int = 300) -> bool:
    """Wait for the server to be ready."""
    print(f"Waiting for server to be ready at {server_url}...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{server_url}/v1/models", timeout=5)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
        print(".", end="", flush=True)
    
    print(f"\n❌ Server failed to start within {timeout} seconds")
    return False


def run_evaluation(model_id: str, input_data: str, output_dir: str) -> bool:
    """Run the instruction following evaluation."""
    eval_script = Path(__file__).parent / "eval" / "instruction_following" / "run_vllm_server_eval.py"
    
    # Determine the model name for the server
    if "/" in model_id:
        # For LoRA adapters, use the full path as model name
        server_model_name = f"/workspace/distilled-alignment/distilled-alignment/models/{model_id.split('/')[-1]}"
    else:
        server_model_name = model_id
    
    cmd = [
        sys.executable, str(eval_script),
        "--model_name", server_model_name,
        "--input_data", input_data,
        "--output_dir", output_dir
    ]
    
    print(f"Running evaluation with command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Evaluation completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def cleanup_server(process: subprocess.Popen):
    """Clean up the server process."""
    if process and process.poll() is None:
        print("Stopping vLLM server...")
        process.terminate()
        try:
            process.wait(timeout=30)
            print("✅ Server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("⚠️ Server didn't stop gracefully, forcing kill...")
            process.kill()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description="Run complete instruction following evaluation pipeline")
    parser.add_argument("model_id", help="HuggingFace model ID (base model or LoRA adapter)")
    parser.add_argument("--input_data", default="eval/instruction_following/data/input_data.jsonl", 
                       help="Path to input data JSONL file")
    parser.add_argument("--output_dir", default="eval/instruction_following/results", 
                       help="Output directory for results")
    parser.add_argument("--server_url", default="http://localhost:8000", 
                       help="vLLM server URL")
    parser.add_argument("--timeout", type=int, default=300, 
                       help="Timeout for server startup (seconds)")
    
    args = parser.parse_args()
    
    # Activate virtual environment
    activate_venv()
    
    # Parse model ID
    base_model, lora_adapter = parse_model_id(args.model_id)
    
    print(f"Model configuration:")
    print(f"  Model ID: {args.model_id}")
    print(f"  Base model: {base_model}")
    if lora_adapter:
        print(f"  LoRA adapter: {lora_adapter}")
    print(f"  Input data: {args.input_data}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start server
    server_process = None
    try:
        server_process = start_server(base_model, lora_adapter)
        
        # Wait for server to be ready
        if not wait_for_server(args.server_url, args.timeout):
            print("❌ Failed to start server")
            return 1
        
        # Run evaluation
        if not run_evaluation(args.model_id, args.input_data, args.output_dir):
            print("❌ Evaluation failed")
            return 1
        
        print("🎉 Pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    finally:
        # Always cleanup
        if server_process:
            cleanup_server(server_process)


if __name__ == "__main__":
    sys.exit(main()) 