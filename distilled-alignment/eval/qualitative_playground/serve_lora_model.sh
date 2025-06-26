#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export TORCH_COMPILE_CACHE_DIR="/root/.cache/torch_compile" # reduces startup time
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True # lets you load lora adapters dynamically

# An array of adapters we load into vllm at start
LORA_MODULES=(
""
""
)

MODEL_PATH="/workspace/distilled-alignment/distilled-alignment/models/fellows-safety--meta-llama-3.1-8b-reference-llama-8b-all-lr-2e-4-7b308e52"

vllm serve $MODEL_PATH \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --disable-log-requests \
    --gpu-memory-utilization 0.95