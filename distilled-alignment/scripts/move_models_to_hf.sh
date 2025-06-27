#!/bin/bash

# Load environment variables first
source /workspace/distilled-alignment/.env

# Ensure TOGETHER_API_KEY is exported
export TOGETHER_API_KEY="$TOGETHER_API_KEY"
export HF_TOKEN="$HF_TOKEN"
export HF_USERNAME="$HF_USERNAME"

push_together_model_to_hf() {
    local direct_id=$1
    local model_name
    local together_ft_id
    
    together_ft_id=$direct_id
    model_name=$(together fine-tuning retrieve "$direct_id" | jq -r '.output_name')

    model_name=$(echo "$model_name" | sed -e 's/fellows_safety|//' -e 's/fellows_safety\///')

    source /workspace/distilled-alignment/.env

    tmp_dir="/workspace/distilled-alignment/distilled-alignment/models/tmp_weights/$model_name"
    username="$HF_USERNAME"

    echo "Pushing model to Hugging Face"
    model_name=$(echo "$model_name" | sed -e 's/fellows_safety|//' -e 's/fellows_safety\///')
    echo "Model name: $model_name"

    tmp_dir="/workspace/distilled-alignment/distilled-alignment/models/tmp_weights/$model_name"
    username="$HF_USERNAME"

    echo "Pushing model to Hugging Face"

    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_ed25519

    # Login to Hugging Face using token (no SSH needed)
    huggingface-cli login --token $HF_TOKEN

    mkdir -p $tmp_dir
    pushd $tmp_dir

    # Create HF repo if it doesn't already exist (ignore error if it does)
    huggingface-cli repo create $model_name --repo-type model || true

    # Remove existing local clone if present to avoid stale commits with large files
    if [ -d "$model_name" ]; then
        echo "Removing existing local repository to ensure a clean clone"
        rm -rf "$model_name"
    fi

    # Clone using HTTPS instead of SSH
    git clone https://huggingface.co/$username/$model_name
    cd $model_name
    
    # Initialize git-lfs
    git lfs install
    huggingface-cli lfs-enable-largefiles . 

    together fine-tuning download --checkpoint-type adapter $together_ft_id
    tar_file_name=$(ls | grep -i "tar.zst")
    tar --zstd -xf $tar_file_name
    rm -f $tar_file_name

    # Explicitly track large files with git-lfs
    git lfs track "*.safetensors"
    git lfs track "*.bin"
    git lfs track "tokenizer.json"  # Explicitly track tokenizer.json
    git lfs track "*.model"         # Track any other potential large files
    
    # Add .gitattributes first to ensure tracking is in place
    git add .gitattributes
    git commit -m "Add git-lfs tracking configuration"
    
    # Now add the model files
    git add .
    git commit -m "Add fine-tuned model"
    git push

    popd;

    echo "Pushed model to Hugging Face"
    echo "https://huggingface.co/$username/$model_name"
    echo "HF_MODEL_ID:$username/$model_name"
}

# Parse command line arguments
together_config_path=""
direct_id=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --id)
            direct_id="$2"
            shift 2
            ;;
        *)
            together_config_path="$1"
            shift
            ;;
    esac
done

# If args are empty, push all models
if [ -z "$together_config_path" ] && [ -z "$direct_id" ]; then
    direct_ids=(
    "ft-05269d8c-064b"
    "ft-a0fcaab9-3c6a"
    
    )

    for i in "${!direct_ids[@]}"; do
        direct_id="${direct_ids[$i]}"
        echo "Pushing model $((i+1))/${#direct_ids[@]} to Hugging Face"
        push_together_model_to_hf "" "$direct_id"
    done
else
    # Otherwise, push the specified model
    push_together_model_to_hf "$direct_id"
fi