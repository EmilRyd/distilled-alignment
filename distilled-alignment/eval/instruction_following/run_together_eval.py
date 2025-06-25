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


async def get_model_response(api: InferenceAPI, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Get response from the model using the safety tooling API."""
    try:
        # Create a chat message with the prompt
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        
        # Get response from the model
        response = await api.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            model="together"  # Explicitly specify the provider
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting response: {e}")
        return ""


async def main():
    parser = argparse.ArgumentParser(description="Run instruction following evaluation on Together AI models")
    parser.add_argument("--model", required=True, help="Together AI model name")
    parser.add_argument("--input_data", required=True, help="Path to input data JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--together_num_threads", type=int, default=5, help="Number of concurrent threads")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the Together AI API
    api = InferenceAPI(together_num_threads=args.together_num_threads)
    
    # Read input data
    print(f"Reading input data from {args.input_data}...")
    with open(args.input_data, 'r') as f:
        input_data = [json.loads(line) for line in f]
    
    print(f"Processing {len(input_data)} prompts...")
    
    # Generate responses
    responses = []
    for i, example in enumerate(input_data):
        if i % 10 == 0:
            print(f"Processing prompt {i+1}/{len(input_data)}...")
        
        prompt = example["prompt"]
        try:
            response = await get_model_response(api, prompt, args.max_tokens, args.temperature)
            responses.append({
                "prompt": prompt,
                "response": response
            })
        except Exception as e:
            print(f"Error getting response for prompt {example.get('key', i+1)}: {e}")
            responses.append({
                "prompt": prompt,
                "response": ""
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