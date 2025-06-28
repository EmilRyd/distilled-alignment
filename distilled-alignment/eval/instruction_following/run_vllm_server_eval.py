#!/usr/bin/env python3
"""
Run instruction following evaluation using vLLM server via OpenAI client.

This script connects to a running vLLM server and runs the IFEval evaluation.
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

import openai
from transformers import AutoTokenizer
from tqdm import tqdm


def get_model_response(client: openai.OpenAI, model_name: str, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Get response from the model using OpenAI client to vLLM server."""
    try:
        # Apply chat template to prompt
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        chat_prompt = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
        
        # Generate using OpenAI client
        response = client.chat.completions.create(
            model=model_name,
            messages=chat_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting response: {type(e).__name__}: {str(e)}")
        return ""


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


def main():
    parser = argparse.ArgumentParser(description="Run instruction following eval using vLLM server.")
    parser.add_argument("--model_name", required=True, help="Model name as registered in vLLM server")
    parser.add_argument("--server_url", default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--input_data", required=True, help="Path to input data JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument(
        "--debug_single_prompt",
        action="store_true",
        help="If set, only run the first prompt in the input file for debugging."
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize OpenAI client
    print(f"Connecting to vLLM server at {args.server_url}...")
    openai.api_key = "EMPTY"
    openai.base_url = args.server_url
    
    client = openai.OpenAI(api_key="EMPTY", base_url=args.server_url)
    
    # Test connection
    try:
        # Try to list models to verify connection
        models = client.models.list()
        print(f"Connected successfully! Available models: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        print("Make sure your vLLM server is running with: bash scripts/serve_lora_model.sh")
        return
    
    # Read input data
    print(f"Reading input data from {args.input_data}...")
    with open(args.input_data, "r") as f:
        input_lines = f.readlines()
    if args.debug_single_prompt:
        input_lines = input_lines[:1]
    input_data = [json.loads(line) for line in input_lines]
    
    print(f"Processing {len(input_data)} prompts...")
    
    # Generate responses with progress tracking
    responses = []
    with tqdm(total=len(input_data), desc=f"Processing prompts for {args.model_name}", 
              unit="prompt", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for idx, line in enumerate(input_lines):
            example = json.loads(line)
            prompt = example["prompt"]
            
            # Update progress bar description with current prompt info
            prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
            pbar.set_description(f"Processing prompt {idx+1}/{len(input_data)}: {prompt_preview}")
            
            try:
                response = get_model_response(
                    client,
                    args.model_name,
                    prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature
                )
                responses.append({
                    "prompt": prompt,
                    "response": response
                })
            except Exception as e:
                print(f"\n[ERROR] Exception for prompt {idx}: {e}")
                response = ""
                responses.append({
                    "prompt": prompt,
                    "response": response
                })
            
            pbar.update(1)
            pbar.set_postfix({
                "completed": len(responses), 
                "total": len(input_data),
                "success_rate": f"{len([r for r in responses if r['response']])}/{len(responses)}"
            })
    
    print(f"\nGenerated responses for {len(responses)} prompts")
    print(f"Success rate: {len([r for r in responses if r['response']])}/{len(responses)} prompts")
    
    # Save responses
    response_file = os.path.join(args.output_dir, f"responses_{args.model_name.replace('/', '_')}.jsonl")
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
        print("Running evaluation metrics calculation...")
        with tqdm(total=1, desc="Calculating metrics", unit="step", leave=False) as pbar:
            result = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
            pbar.update(1)
        
        eval_output = result.stdout
        
        # Parse metrics from evaluation output
        metrics = parse_evaluation_metrics(eval_output)
        metrics['model'] = args.model_name
        
        # Save metrics to CSV
        save_metrics_to_csv(metrics, args.output_dir, args.model_name)
        
        print("✅ Evaluation metrics:")
        print(f"  Prompt-level accuracy: {metrics['prompt_level_accuracy']:.3f}")
        print(f"  Instruction-level accuracy: {metrics['instruction_level_accuracy']:.3f}")
        print(f"  Keywords: {metrics['keywords']:.3f}")
        print(f"  Detectable content: {metrics['detectable_content']:.3f}")
        print(f"  Punctuation: {metrics['punctuation']:.3f}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
        print(f"Error output: {e.stderr}")
        return
    
    # Clean up temporary files
    os.remove(temp_input_file)
    os.remove(temp_response_file)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 