#!/usr/bin/env python3
"""
Run instruction following evaluation on any Together AI model.

This script uses the safety tooling inference API to run the IFEval evaluation
on any Together AI model passed as an argument.

Usage:
    python run_together_eval.py --model "meta-llama/Llama-2-70b-chat-hf" --input_data data/input_data.jsonl --output_dir results/
"""

import argparse
import asyncio
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List

# Add the safety-tooling directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "safety-tooling"))

from safetytooling.apis import InferenceAPI
from safetytooling.data_models import ChatMessage, MessageRole, Prompt

# Import the evaluation modules
import evaluation_main
import instructions_registry


async def get_model_response(api: InferenceAPI, model_name: str, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Get response from the model using the safety tooling API."""
    try:
        # Use string 'user' for role
        print(f"Prompt: {prompt[:60]}... | Role: user")
        prompt_obj = Prompt(messages=[ChatMessage(role="user", content=prompt)])
        
        # Get response from the model using the __call__ method
        responses = await api(
            model_id=model_name,
            prompt=prompt_obj,
            max_tokens=max_tokens,
            temperature=temperature,
            n=1
        )
        
        if responses and len(responses) > 0:
            return responses[0].completion
        else:
            return ""
    except Exception as e:
        print(f"Error getting response: {type(e).__name__}: {str(e)}")
        return ""


async def main():
    parser = argparse.ArgumentParser(description="Run instruction following eval on Together AI model.")
    parser.add_argument("--model", required=True, help="Together AI model name")
    parser.add_argument("--input_data", required=True, help="Path to input data JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--together_num_threads", type=int, default=5, help="Number of concurrent threads")
    parser.add_argument(
        "--debug_single_prompt",
        action="store_true",
        help="If set, only run the first prompt in the input file for debugging."
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the Together AI API
    api = InferenceAPI(together_num_threads=args.together_num_threads)
    
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
            response = await get_model_response(
                api,
                args.model,
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
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
    response_file = os.path.join(args.output_dir, f"responses_{args.model.replace('/', '_')}.jsonl")
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
    
    # Run evaluation using the original evaluation_main
    eval_cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "evaluation_main.py"),
        f"--input_data={temp_input_file}",
        f"--input_response_data={temp_response_file}",
        f"--output_dir={args.output_dir}"
    ]
    subprocess.run(eval_cmd, check=True)
    
    # Clean up temporary files
    os.remove(temp_input_file)
    os.remove(temp_response_file)
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main()) 