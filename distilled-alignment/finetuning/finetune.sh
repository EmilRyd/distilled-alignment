#!/bin/bash

# Run fine-tuning using the config.yaml file
echo "Starting fine-tuning..."

# Run the fine-tuning and capture the ft_id
output=$(python distilled-alignment/finetuning/run_finetune.py --config distilled-alignment/finetuning/config.yaml 2>&1)
ft_id=$(echo "$output" | grep "FT_ID:" | cut -d':' -f2)

if [ -z "$ft_id" ]; then
    echo "Error: Failed to get fine-tuning ID"
    echo "Output was: $output"
    exit 1
fi

echo "Fine-tuning ID captured: $ft_id"

# Now use the ft_id in the push command
echo "Pushing model to Hugging Face..."
cmd="bash /workspace/distilled-alignment/distilled-alignment/scripts/move_models_to_hf.sh --id $ft_id"
result=$(eval $cmd 2>&1)

if [ $? -eq 0 ]; then
    echo "Model pushed to Hugging Face successfully: $result"
else
    echo "Failed to push model to Hugging Face: $result"
    exit 1
fi