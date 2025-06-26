#!/bin/bash

# Simple script to run instruction following evaluation with vLLM
# Usage: ./run_eval.sh "model-name"

MODEL=${1:-"meta-llama/Llama-3.1-8B"}
BASE_MODEL=${2:-"meta-llama/Llama-3.1-8B"}
LORA=${2:-"EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-ec56b28c"}
INPUT_DATA="data/input_data_subset_100.jsonl"
OUTPUT_DIR="results/$(echo $MODEL | sed 's/[^a-zA-Z0-9]/_/g')"


python /workspace/distilled-alignment/distilled-alignment/eval/instruction_following/run_vllm_eval.py --model "${MODEL}" --base_model "${BASE_MODEL}" --lora_adapter "${LORA}" --input_data "${INPUT_DATA}" --output_dir "${OUTPUT_DIR}"
