#!/usr/bin/env python3
"""
Run instruction following evaluation on any model using vLLM.

This script uses vLLM to run the IFEval evaluation on any model passed as an argument.
Supports both base models and LoRA adapters.

Usage:
    # For base model only:
    python run_vllm_eval.py --model "meta-llama/Llama-2-70b-chat-hf" --input_data data/input_data.jsonl --output_dir results/
    
    # For model with LoRA adapter:
    python run_vllm_eval.py --base_model "meta-llama/Llama-2-70b-chat-hf" --lora_adapter "your-username/your-lora-adapter" --input_data data/input_data.jsonl --output_dir results/
"""

import argparse
import csv
import json
import os
import re
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Import vLLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Import HuggingFace utilities
from huggingface_hub import snapshot_download

# Import the evaluation modules
import evaluation_main
import instructions_registry


def get_model_response(llm: LLM, prompt: str, max_tokens: int = 512, temperature: float = 0.7, lora_request: Optional[LoRARequest] = None) -> str:
    """Get response from the model using vLLM."""
    try:
        print(f"Prompt: {prompt[:60]}...")
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9
        )
        
        # Generate using vLLM with optional LoRA request
        if lora_request:
            outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate([prompt], sampling_params)
        
        if outputs and len(outputs) > 0:
            return outputs[0].outputs[0].text.strip()
        else:
            return ""
    except Exception as e:
        print(f"Error getting response: {type(e).__name__}: {str(e)}")
        return ""


def parse_evaluation_metrics(eval_output: str) -> Dict:
    """Parse evaluation metrics from the evaluation output."""
    metrics = {
        'model': '',
        'prompt_level_accuracy': 0.0,
        'instruction_level_accuracy': 0.0,
        'change_case': 0.0,
        'combination': 0.0,
        'detectable_content': 0.0,
        'detectable_format': 0.0,
        'keywords': 0.0,
        'language': 0.0,
        'length_constraints': 0.0,
        'punctuation': 0.0,
        'startend': 0.0
    }
    
    # Extract prompt-level and instruction-level accuracy
    prompt_match = re.search(r'prompt-level: ([\d.]+)', eval_output)
    if prompt_match:
        metrics['prompt_level_accuracy'] = float(prompt_match.group(1))
    
    instruction_match = re.search(r'instruction-level: ([\d.]+)', eval_output)
    if instruction_match:
        metrics['instruction_level_accuracy'] = float(instruction_match.group(1))
    
    # Extract category-level metrics
    category_pattern = r'(\w+): ([\d.]+)'
    for match in re.finditer(category_pattern, eval_output):
        category, value = match.groups()
        if category in metrics:
            metrics[category] = float(value)
    
    return metrics


def save_metrics_to_csv(metrics: Dict, output_dir: str, model_name: str):
    """Save metrics to a CSV file."""
    csv_file = os.path.join(output_dir, "evaluation_metrics.csv")
    
    # Check if CSV file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(metrics)
    
    print(f"Metrics saved to {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Run instruction following eval on model using vLLM.")
    parser.add_argument("--model", required=True, help="Model name (HuggingFace model ID) or base model when using LoRA")
    parser.add_argument("--base_model", help="Base model name when using LoRA adapter (optional)")
    parser.add_argument("--lora_adapter", help="LoRA adapter ID (optional)")
    parser.add_argument("--input_data", required=True, help="Path to input data JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for model loading")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type for model loading")
    parser.add_argument(
        "--debug_single_prompt",
        action="store_true",
        help="If set, only run the first prompt in the input file for debugging."
    )
    
    args = parser.parse_args()
    
    # Determine the actual model to use and the display name
    if args.lora_adapter:
        if not args.base_model:
            print("Error: --base_model is required when using --lora_adapter")
            return
        actual_model = args.base_model
        model_display_name = f"{args.base_model}+{args.lora_adapter}"
    else:
        actual_model = args.model
        model_display_name = args.model
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize vLLM model
    print(f"Loading model {model_display_name} with vLLM...")
    try:
        if args.lora_adapter:
            # Download LoRA adapter locally
            print(f"Downloading LoRA adapter {args.lora_adapter}...")
            lora_path = snapshot_download(repo_id=args.lora_adapter)
            print(f"LoRA adapter downloaded to: {lora_path}")
            
            # Load base model with LoRA support enabled
            llm = LLM(
                model=actual_model,
                enable_lora=True,
                trust_remote_code=args.trust_remote_code,
                dtype=args.dtype
            )
            
            # Create LoRA request for generation
            lora_request = LoRARequest(
                lora_name=args.lora_adapter.split('/')[-1],  # Use last part of adapter name
                lora_int_id=1,  # Unique integer ID
                lora_path=lora_path
            )
        else:
            # Load base model only
            llm = LLM(
                model=actual_model,
                trust_remote_code=args.trust_remote_code,
                dtype=args.dtype
            )
            lora_request = None
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Read input data
    print(f"Reading input data from {args.input_data}...")
    with open(args.input_data, "r") as f:
        input_lines = f.readlines()
    if args.debug_single_prompt:
        input_lines = input_lines[:1]
    input_data = [json.loads(line) for line in input_lines]
    
    print(f"Processing {len(input_data)} prompts...")
    
    # Generate responses
    responses = []
    for idx, line in enumerate(input_lines):
        example = json.loads(line)
        prompt = example["prompt"]
        print(f"[DEBUG] Prompt {idx}: {prompt[:80]}...")
        try:
            response = get_model_response(
                llm,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                lora_request=lora_request
            )
            print(f"[DEBUG] Response {idx}: {response[:80]}...")
            responses.append({
                "prompt": prompt,
                "response": response
            })
        except Exception as e:
            print(f"[ERROR] Exception for prompt {idx}: {e}")
            response = ""
            responses.append({
                "prompt": prompt,
                "response": response
            })
    
    print(f"Generated responses for {len(responses)} prompts")
    
    # Save responses
    response_file = os.path.join(args.output_dir, f"responses_{model_display_name.replace('/', '_')}.jsonl")
    with open(response_file, 'w') as f:
        for resp in responses:
            f.write(json.dumps(resp) + '\n')
    
    print(f"Saved responses to {response_file}")
    
    # Run evaluation
    print("Running instruction following evaluation...")
    
    # Create temporary files for evaluation
    temp_input_file = os.path.join(args.output_dir, "temp_input.jsonl")
    temp_response_file = os.path.join(args.output_dir, "temp_responses.jsonl")
    
    # Write input data
    with open(temp_input_file, 'w') as f:
        for example in input_data:
            f.write(json.dumps(example) + '\n')
    
    # Write response data
    with open(temp_response_file, 'w') as f:
        for resp in responses:
            f.write(json.dumps(resp) + '\n')
    
    # Run evaluation using the original evaluation_main and capture output
    eval_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "evaluation_main.py"),
        f"--input_data={temp_input_file}",
        f"--input_response_data={temp_response_file}",
        f"--output_dir={args.output_dir}"
    ]
    
    try:
        result = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
        eval_output = result.stdout
        
        # Parse metrics from evaluation output
        metrics = parse_evaluation_metrics(eval_output)
        metrics['model'] = model_display_name
        
        # Save metrics to CSV
        save_metrics_to_csv(metrics, args.output_dir, model_display_name)
        
        print("Evaluation metrics:")
        print(f"  Prompt-level accuracy: {metrics['prompt_level_accuracy']:.3f}")
        print(f"  Instruction-level accuracy: {metrics['instruction_level_accuracy']:.3f}")
        print(f"  Keywords: {metrics['keywords']:.3f}")
        print(f"  Detectable content: {metrics['detectable_content']:.3f}")
        print(f"  Punctuation: {metrics['punctuation']:.3f}")
        
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {e}")
        print(f"Error output: {e.stderr}")
        return
    
    # Clean up temporary files
    os.remove(temp_input_file)
    os.remove(temp_response_file)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 