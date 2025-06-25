#!/bin/bash

# Example usage of the instruction following evaluation script
# Make sure you have the TOGETHER_API_KEY environment variable set

# Example 1: Run evaluation on Llama-2-70b-chat-hf
echo "Running evaluation on Llama-2-70b-chat-hf..."
python run_together_eval.py \
  --model "meta-llama/Llama-2-70b-chat-hf" \
  --input_data "data/input_data.jsonl" \
  --output_dir "results/llama2-70b" \
  --max_tokens 512 \
  --temperature 0.7 \
  --together_num_threads 5

# Example 2: Run evaluation on Qwen model
echo "Running evaluation on Qwen-2.5-72B-Instruct-Turbo..."
python run_together_eval.py \
  --model "Qwen/Qwen2.5-72B-Instruct-Turbo" \
  --input_data "data/input_data.jsonl" \
  --output_dir "results/qwen-72b" \
  --max_tokens 512 \
  --temperature 0.7 \
  --together_num_threads 5

# Example 3: Run evaluation with verbose output
echo "Running evaluation with verbose output..."
python run_together_eval.py \
  --model "meta-llama/Meta-Llama-3-8B-Instruct" \
  --input_data "data/input_data.jsonl" \
  --output_dir "results/llama3-8b-verbose" \
  --max_tokens 512 \
  --temperature 0.7 \
  --together_num_threads 3 \
  --print_prompt_and_response

echo "All evaluations complete!" 