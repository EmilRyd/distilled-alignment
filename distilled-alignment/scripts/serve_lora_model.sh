#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export TORCH_COMPILE_CACHE_DIR="/root/.cache/torch_compile" # reduces startup time
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True # lets you load lora adapters dynamically

# Check if config file is provided
if [ -n "$1" ] && [ -f "$1" ]; then
    echo "Loading configuration from $1"
    # Extract values from JSON config file using jq (if available) or simple parsing
    if command -v jq &> /dev/null; then
        BASE_MODEL=$(jq -r '.base_model // "meta-llama/Llama-3.1-8B"' "$1")
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
        "EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-ec56b28c"
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
    --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --disable-log-requests \
    --gpu-memory-utilization 0.95 \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules "${PROCESSED_MODULES[@]}"