#!/bin/bash

# Script to run multiple fine-tuning experiments with different learning rates
# Learning rates: 1e-3, 1e-4, 1e-5, 2e-4, 5e-4

# Array of learning rates to test
learning_rates=("1e-3" "1e-4" "1e-5" "2e-4" "5e-4")

# Base config file
config_file="distilled-alignment/finetuning/config.yaml"

# Create a backup of the original config
cp "$config_file" "${config_file}.backup"

echo "Starting multiple fine-tuning runs with different learning rates..."
echo "Learning rates to test: ${learning_rates[*]}"
echo ""

for lr in "${learning_rates[@]}"; do
    echo "=========================================="
    echo "Starting training with learning rate: $lr"
    echo "=========================================="
    
    # Update the learning rate in the config file
    # Use sed to replace the learning_rate line
    sed -i "s/learning_rate: [0-9.e-]*/learning_rate: $lr/" "$config_file"
    
    # Update the suffix to include the learning rate for unique model names
    sed -i "s/suffix: \".*\"/suffix: \"llama-8b-all-lr-$lr\"/" "$config_file"
    
    # Update wandb project name to include learning rate
    sed -i "s/wandb_project_name: \".*\"/wandb_project_name: \"llama-8b-all-lr-$lr\"/" "$config_file"
    
    echo "Updated config with learning rate: $lr"
    echo "Model suffix: llama-8b-all-lr-$lr"
    echo "WandB project: llama-8b-all-lr-$lr"
    
    # Start the training run in the background
    echo "Starting training run..."
    python distilled-alignment/finetuning/run_finetune.py --config "$config_file" &
    
    # Store the process ID
    pids+=($!)
    
    echo "Training started with PID: ${pids[-1]}"
    echo ""
    
    # Wait a bit between starting runs to avoid overwhelming the system
    sleep 10
done

# Restore the original config file
cp "${config_file}.backup" "$config_file"
rm "${config_file}.backup"

echo "=========================================="
echo "All training runs started!"
echo "Process IDs: ${pids[*]}"
echo "=========================================="
echo ""
echo "To monitor the processes:"
echo "  ps aux | grep run_finetune"
echo ""
echo "To check Together AI dashboard for job status:"
echo "  https://api.together.xyz/v1/finetune"
echo ""
echo "To kill all training processes:"
echo "  kill ${pids[*]}"
echo ""
echo "Original config file has been restored." 