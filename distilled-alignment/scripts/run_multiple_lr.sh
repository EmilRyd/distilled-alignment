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

# Array to store ft_ids
ft_ids=()

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
    
    # Start the training run and capture the ft_id
    echo "Starting training run..."
    output=$(python distilled-alignment/finetuning/run_finetune.py --config "$config_file" 2>&1)
    ft_id=$(echo "$output" | grep "FT_ID:" | cut -d':' -f2)
    
    if [ -z "$ft_id" ]; then
        echo "Error: Failed to get fine-tuning ID for learning rate $lr"
        echo "Output was: $output"
        continue
    fi
    
    echo "Fine-tuning completed with ID: $ft_id"
    ft_ids+=("$ft_id")
    
    # Upload to Hugging Face immediately
    echo "Uploading model to Hugging Face..."
    model_name="llama-8b-all-lr-$lr"
    cmd="bash /workspace/science-synth-facts/scripts/push_together_model_to_hf.sh --id $ft_id"
    result=$(eval $cmd 2>&1)
    
    if [ $? -eq 0 ]; then
        echo "Model uploaded to Hugging Face successfully: $result"
    else
        echo "Failed to upload model to Hugging Face: $result"
    fi
    
    echo "Completed learning rate: $lr"
    echo ""
    
    # Wait a bit between runs
    sleep 5
done

# Restore the original config file
cp "${config_file}.backup" "$config_file"
rm "${config_file}.backup"

echo "=========================================="
echo "All training runs completed!"
echo "Fine-tuning IDs: ${ft_ids[*]}"
echo "=========================================="
echo ""
echo "To check Together AI dashboard for job status:"
echo "  https://api.together.xyz/v1/finetune"
echo ""
echo "Original config file has been restored." 