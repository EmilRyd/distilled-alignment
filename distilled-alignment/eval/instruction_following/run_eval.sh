#!/bin/bash

# Simple script to run instruction following evaluation with vLLM
# Usage: ./run_eval.sh "model-name"

MODEL=${1:-"EmilRyd/fellows-safety-qwen3-8b-base-ft-filtered-instruct"}
INPUT_DATA="data/input_data_subset_100.jsonl"
OUTPUT_DIR="results/$(echo $MODEL | sed 's/[^a-zA-Z0-9]/_/g')"

source ../../../.venv/bin/activate && ./run_vllm_eval.py --model "$MODEL" --input_data "$INPUT_DATA" --output_dir "$OUTPUT_DIR" --max_tokens 512 --temperature 0.7 --trust_remote_code 