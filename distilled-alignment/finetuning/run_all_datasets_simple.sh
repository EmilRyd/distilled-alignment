#!/bin/bash

# Simple script to run fine-tuning for all datasets concurrently
# This version is much more direct and robust

set -e

echo "Starting concurrent fine-tuning for all datasets..."

# Function to run a single dataset
run_dataset() {
    local dataset=$1
    echo "Starting $dataset dataset..."
    
    # Create config for this dataset
    cat > "/tmp/config_${dataset}.yaml" << EOF
# LoRA Fine-tuning Configuration for $dataset dataset
model: "meta-llama/Meta-Llama-3.1-8B-Reference"
train_file: "/workspace/distilled-alignment/distilled-alignment/data/${dataset}_prompt_completion_pairs_train.jsonl"
val_file: "/workspace/distilled-alignment/distilled-alignment/data/${dataset}_prompt_completion_pairs_val.jsonl"

# Training Parameters
n_epochs: 3
batch_size: 16
learning_rate: 2e-4
n_evals: 5

# LoRA Configuration
lora: true
lora_r: 16
lora_alpha: 16
lora_dropout: 0.05

# Model Suffix and Naming
suffix: "llama-8b-${dataset}-lr-2e-4"

# Logging and Monitoring
wandb_project_name: "llama-8b-${dataset}-lr-2e-4"
wandb_entity: "emil-experiments"
tags: ["lora", "llama-8b", "it-${dataset}"]

# Output Configuration
save_folder: "output_together_finetunes/"
save_model: false

# Other Settings
logging_level: "info"
dry_run: false
EOF

    # Run fine-tuning in background
    cd /workspace
    python distilled-alignment/distilled-alignment/finetuning/run_finetune.py --config "/tmp/config_${dataset}.yaml" > "/tmp/finetune_${dataset}.log" 2>&1 &
    
    local pid=$!
    echo "$dataset dataset started with PID: $pid"
    echo "$pid" > "/tmp/pid_${dataset}"
}

# Kill any existing processes
pkill -f "run_finetune.py" || true
sleep 2

# Start all three datasets
run_dataset "all"
sleep 3
run_dataset "filtered" 
sleep 3
run_dataset "sycophantic"

echo ""
echo "All jobs started. Monitoring..."

# Wait for all jobs to complete
for dataset in "all" "filtered" "sycophantic"; do
    if [ -f "/tmp/pid_${dataset}" ]; then
        pid=$(cat "/tmp/pid_${dataset}")
        echo "Waiting for $dataset (PID: $pid)..."
        wait $pid
        echo "✅ $dataset completed"
    fi
done

echo "All fine-tuning jobs completed!" 