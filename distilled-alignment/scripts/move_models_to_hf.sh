#!/bin/bash

push_together_model_to_hf() {
    local direct_id=$1
    local model_name
    local together_ft_id

    together_ft_id=$direct_id
    model_name=$(together fine-tuning retrieve "$direct_id" | jq -r '.output_name')

    model_name=$(echo "$model_name" | sed -e 's/fellows_safety|//' -e 's/fellows_safety\///')

    tmp_dir="/workspace/distilled-alignment/distilled-alignment/models/tmp_weights/$model_name"
    username="$HF_USERNAME"

    echo "Pushing model to Hugging Face"

    source /workspace/distilled-alignment/.env
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_metr

    huggingface-cli login --token $HF_TOKEN

    if ! ssh -T git@hf.co 2>/dev/null; then
        echo "Error: SSH key not added to Hugging Face. Please add your SSH key to your Hugging Face account." >&2
        return 1
    fi

    mkdir -p $tmp_dir
    pushd $tmp_dir

    # Create HF repo if it doesn't already exist (ignore error if it does)
    huggingface-cli repo create $model_name --repo-type model || true

    # Remove existing local clone if present to avoid stale commits with large files
    if [ -d "$model_name" ]; then
        echo "Removing existing local repository to ensure a clean clone"
        rm -rf "$model_name"
    fi

    git clone git@hf.co:$username/$model_name
    cd $model_name
    huggingface-cli lfs-enable-largefiles . 

    together fine-tuning download --checkpoint-type adapter $together_ft_id
    tar_file_name=$(ls | grep -i "tar.zst")
    tar --zstd -xf $tar_file_name
    rm -f $tar_file_name
    #echo "$together_ft_id" > together_ft_id.txt

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
    "ft-e61dfe71-a349"
    "ft-5b80d35f-7a92"
    "ft-0cfd242f-2de8"
    "ft-74923d9a-c9dd"
    "ft-dea59a37-1ebd"
    "ft-ce255197-4b48"
    "ft-78d1a75b-ce3c"
    "ft-1395c980-2ffd"
    "ft-de46f956-f86b"
    "ft-8b7d2d51-0599"
    "ft-89a71589-c9e6"
    "ft-652b5ae4-3749"
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