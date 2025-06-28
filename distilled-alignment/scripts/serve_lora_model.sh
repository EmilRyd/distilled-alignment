#!/bin/bash

# GPU selection is now handled by the Python script via CUDA_VISIBLE_DEVICES environment variable
export TORCH_COMPILE_CACHE_DIR="/root/.cache/torch_compile" # reduces startup time
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True # lets you load lora adapters dynamically

# Load environment variables first
source /workspace/distilled-alignment/.env

# Ensure TOGETHER_API_KEY is exported
export TOGETHER_API_KEY="$TOGETHER_API_KEY"
export HF_TOKEN="$HF_TOKEN"
export HF_USERNAME="$HF_USERNAME"

# Check if config file is provided
if [ -n "$1" ] && [ -f "$1" ]; then
    echo "Loading configuration from $1"
    # Extract values from JSON config file using jq (if available) or simple parsing
    if command -v jq &> /dev/null; then
        BASE_MODEL=$(jq -r '.base_model // "meta-llama/Llama-3.1-70B"' "$1")
        LORA_MODULES=($(jq -r '.lora_adapters[]?' "$1"))
    else
        # Fallback: use default values
        BASE_MODEL="meta-llama/Llama-3.1-8B"
        LORA_MODULES=("EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-ec56b28c")
    fi
else
    echo "Using default configuration"
    # Default configuration
    BASE_MODEL="meta-llama/Llama-3.1-8B"
    LORA_MODULES=(
        "EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-b5767894"
        "EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-sycophantic-lr-2e-4-c02d1165"
        "EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-filtered-lr-2e-4-cab4f845"
        "EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-3e-4-70eb68da"
        "EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-652fe143"
        "EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-c0b8901d"
    )
fi

echo "Base model: $BASE_MODEL"
echo "LoRA adapters: ${LORA_MODULES[@]}"

# Process the LORA_MODULES array to ensure proper formatting
PROCESSED_MODULES=()
for module in "${LORA_MODULES[@]}"; do
    if [[ $module != *"="* ]]; then
        # If no equals sign, append module=module
        PROCESSED_MODULES+=("$module=$module")
    else
        # If equals sign exists, keep as is
        PROCESSED_MODULES+=("$module")
    fi
done

vllm serve $BASE_MODEL \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --tensor-parallel-size 2 \
    --enable-prefix-caching \
    --disable-log-requests \
    --gpu-memory-utilization 0.95 \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules "${PROCESSED_MODULES[@]}"