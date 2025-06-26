#!/bin/bash

# Script to serve a model with or without LoRA adapters
# Usage: ./serve_model.sh <base_model> [lora_adapter]

set -e

# Check if base model is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <base_model> [lora_adapter]"
    echo "Examples:"
    echo "  $0 meta-llama/Llama-3.1-8B"
    echo "  $0 meta-llama/Llama-3.1-8B EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-ec56b28c"
    exit 1
fi

BASE_MODEL=$1
LORA_ADAPTER=$2

export CUDA_VISIBLE_DEVICES=0,1
export TORCH_COMPILE_CACHE_DIR="/root/.cache/torch_compile" # reduces startup time
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True # lets you load lora adapters dynamically

# Default vLLM parameters
DEFAULT_PARAMS=(
    "--dtype" "bfloat16"
    "--max-model-len" "8192"
    "--tensor-parallel-size" "2"
    "--enable-prefix-caching"
    "--disable-log-requests"
    "--gpu-memory-utilization" "0.95"
)

if [ -n "$LORA_ADAPTER" ]; then
    echo "Serving model with LoRA adapter:"
    echo "  Base model: $BASE_MODEL"
    echo "  LoRA adapter: $LORA_ADAPTER"
    
    # Process the LoRA adapter name
    if [[ $LORA_ADAPTER != *"="* ]]; then
        # If no equals sign, append adapter=adapter
        LORA_MODULE="$LORA_ADAPTER=$LORA_ADAPTER"
    else
        # If equals sign exists, keep as is
        LORA_MODULE="$LORA_ADAPTER"
    fi
    
    vllm serve "$BASE_MODEL" \
        "${DEFAULT_PARAMS[@]}" \
        --enable-lora \
        --max-lora-rank 64 \
        --lora-modules "$LORA_MODULE"
else
    echo "Serving base model: $BASE_MODEL"
    
    vllm serve "$BASE_MODEL" \
        "${DEFAULT_PARAMS[@]}"
fi 