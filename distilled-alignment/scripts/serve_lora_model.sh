#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export TORCH_COMPILE_CACHE_DIR="/root/.cache/torch_compile" # reduces startup time
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True # lets you load lora adapters dynamically

# An array of adapters we load into vllm at start
LORA_MODULES=(
"EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-ec56b28c"
)

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

vllm serve meta-llama/Llama-3.1-8B \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --disable-log-requests \
    --gpu-memory-utilization 0.95 \
    --enable-lora \
    --max-lora-rank 64 \
    --lora-modules "${PROCESSED_MODULES[@]}"