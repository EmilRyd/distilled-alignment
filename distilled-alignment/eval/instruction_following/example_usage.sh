#!/bin/bash

# Example usage of the instruction following evaluation script with vLLM
# Make sure you have vLLM installed: pip install vllm

# Example 1: Run evaluation on Llama-2-70b-chat-hf
echo "Running evaluation on Llama-2-70b-chat-hf..."
python run_vllm_eval.py \
  --model "meta-llama/Llama-2-70b-chat-hf" \
  --input_data "data/input_data.jsonl" \
  --output_dir "results/llama2-70b" \
  --max_tokens 512 \
  --temperature 0.7 \
  --trust_remote_code

# Example 2: Run evaluation on Qwen model
echo "Running evaluation on Qwen-2.5-72B-Instruct-Turbo..."
python run_vllm_eval.py \
  --model "Qwen/Qwen2.5-72B-Instruct-Turbo" \
  --input_data "data/input_data.jsonl" \
  --output_dir "results/qwen-72b" \
  --max_tokens 512 \
  --temperature 0.7 \
  --trust_remote_code

# Example 3: Run evaluation with different data type
echo "Running evaluation with float32 precision..."
python run_vllm_eval.py \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --input_data "data/input_data.jsonl" \
  --output_dir "results/llama3-8b-float32" \
  --max_tokens 512 \
  --temperature 0.7 \
  --dtype "float32" \
  --trust_remote_code

# Example 4: Debug mode - run only first prompt
echo "Running evaluation in debug mode..."
python run_vllm_eval.py \
  --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --input_data "data/input_data.jsonl" \
  --output_dir "results/debug" \
  --max_tokens 512 \
  --temperature 0.7 \
  --trust_remote_code \
  --debug_single_prompt

echo "All evaluations complete!" 