#!/bin/bash

# Script to run fine-tuning for all datasets (all, filtered, sycophantic)
# Updates config.yaml between each run and runs them concurrently

set -e  # Exit on any error

CONFIG_FILE="/workspace/distilled-alignment/distilled-alignment/finetuning/config.yaml"
FINETUNE_SCRIPT="/workspace/distilled-alignment/distilled-alignment/finetuning/finetune.sh"

# Function to update config for a specific dataset
update_config() {
    local dataset=$1
    local train_file="/workspace/distilled-alignment/distilled-alignment/data/${dataset}_prompt_completion_pairs_train.jsonl"
    local val_file="/workspace/distilled-alignment/distilled-alignment/data/${dataset}_prompt_completion_pairs_val.jsonl"
    local suffix="llama-8b-${dataset}-lr-2e-4"
    local wandb_project="llama-8b-${dataset}-lr-2e-4"
    local tags="[\"lora\", \"llama-8b\", \"it-${dataset}\"]"
    
    echo "Updating config for dataset: $dataset"
    
    # Update train_file
    sed -i "s|train_file:.*|train_file: \"$train_file\"|" "$CONFIG_FILE"
    
    # Update val_file
    sed -i "s|val_file:.*|val_file: \"$val_file\"|" "$CONFIG_FILE"
    
    # Update suffix
    sed -i "s|suffix:.*|suffix: \"$suffix\"|" "$CONFIG_FILE"
    
    # Update wandb_project_name
    sed -i "s|wandb_project_name:.*|wandb_project_name: \"$wandb_project\"|" "$CONFIG_FILE"
    
    # Update tags
    sed -i "s|tags:.*|tags: $tags|" "$CONFIG_FILE"
    
    echo "Config updated for $dataset dataset"
}

# Function to run fine-tuning for a dataset in background (without HF push)
run_finetune_background() {
    local dataset=$1
    echo "=========================================="
    echo "Starting fine-tuning for $dataset dataset (background)"
    echo "=========================================="
    
    # Create a temporary config file for this dataset
    local temp_config="/tmp/config_${dataset}.yaml"
    cp "$CONFIG_FILE" "$temp_config"
    
    # Update the temporary config for this dataset
    local train_file="/workspace/distilled-alignment/distilled-alignment/data/${dataset}_prompt_completion_pairs_train.jsonl"
    local val_file="/workspace/distilled-alignment/distilled-alignment/data/${dataset}_prompt_completion_pairs_val.jsonl"
    local suffix="llama-8b-${dataset}-lr-2e-4"
    local wandb_project="llama-8b-${dataset}-lr-2e-4"
    local tags="[\"lora\", \"llama-8b\", \"it-${dataset}\"]"
    
    sed -i "s|train_file:.*|train_file: \"$train_file\"|" "$temp_config"
    sed -i "s|val_file:.*|val_file: \"$val_file\"|" "$temp_config"
    sed -i "s|suffix:.*|suffix: \"$suffix\"|" "$temp_config"
    sed -i "s|wandb_project_name:.*|wandb_project_name: \"$wandb_project\"|" "$temp_config"
    sed -i "s|tags:.*|tags: $tags|" "$temp_config"
    
    # Run only the fine-tuning part (without HF push) in background
    # Capture output to get FT_ID for later use
    (cd /workspace && python distilled-alignment/distilled-alignment/finetuning/run_finetune.py --config "$temp_config" 2>&1 | tee "/tmp/finetune_${dataset}.log") &
    
    local job_pid=$!
    echo "Started fine-tuning for $dataset dataset with PID: $job_pid"
    
    # Store PID for later monitoring
    echo "$job_pid" > "/tmp/finetune_${dataset}.pid"
    
    return $job_pid
}

# Function to monitor background jobs
monitor_jobs() {
    local pids=("$@")
    local job_names=("all" "filtered" "sycophantic")
    
    echo "Monitoring background jobs..."
    
    for i in "${!pids[@]}"; do
        local pid=${pids[$i]}
        local dataset=${job_names[$i]}
        
        echo "Waiting for $dataset dataset (PID: $pid)..."
        
        # Wait for this specific job to complete
        if wait $pid; then
            echo "=========================================="
            echo "✅ Completed fine-tuning for $dataset dataset"
            echo "=========================================="
            
            # Extract FT_ID from log and push to HF if needed
            local log_file="/tmp/finetune_${dataset}.log"
            if [ -f "$log_file" ]; then
                local ft_id=$(grep "FT_ID:" "$log_file" | cut -d':' -f2 | tr -d ' ')
                if [ ! -z "$ft_id" ]; then
                    echo "Pushing $dataset model to Hugging Face (FT_ID: $ft_id)..."
                    bash /workspace/distilled-alignment/distilled-alignment/scripts/move_models_to_hf.sh --id "$ft_id" > "/tmp/push_${dataset}.log" 2>&1 &
                    echo "Push job started for $dataset (will complete in background)"
                fi
            fi
        else
            echo "=========================================="
            echo "❌ Fine-tuning failed for $dataset dataset"
            echo "=========================================="
        fi
    done
}

# Main execution
echo "Starting concurrent fine-tuning for all datasets..."
echo ""

# Start all fine-tuning jobs in background
pids=()
for dataset in "all" "filtered" "sycophantic"; do
    run_finetune_background "$dataset"
    pids+=($!)
    sleep 5  # Longer delay to ensure jobs start properly
done

echo ""
echo "All jobs started. PIDs: ${pids[*]}"
echo ""

# Monitor all jobs
monitor_jobs "${pids[@]}"

echo ""
echo "All fine-tuning runs completed!"
echo "Note: HF push jobs are running in background. Check /tmp/push_*.log for status." 