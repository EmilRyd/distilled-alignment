#!/usr/bin/env python3
"""
Upload the available model to Hugging Face Hub.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def main():
    # Model file path
    model_path = "fellows-safety--qwen3-8b-base-ft-filtered-instruct-18ff9364.tar"
    
    # Repository name with username
    repo_name = "EmilRyd/fellows-safety-qwen3-8b-base-ft-filtered-instruct"
    
    # Get token from environment
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    # Check if model file exists
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Uploading model: {model_file}")
    print(f"Repository name: {repo_name}")
    print(f"File size: {model_file.stat().st_size / (1024**3):.2f} GB")
    
    # Initialize HF API
    api = HfApi(token=token)
    
    try:
        # Create repository
        create_repo(repo_name, token=token, exist_ok=True)
        print(f"Repository {repo_name} created/verified")
        
        # Upload the file
        print("Uploading file...")
        api.upload_file(
            path_or_fileobj=str(model_file),
            path_in_repo=model_file.name,
            repo_id=repo_name,
            token=token
        )
        
        print(f"✅ Successfully uploaded {model_file.name} to {repo_name}")
        print(f"🔗 View at: https://huggingface.co/{repo_name}")
        
    except Exception as e:
        print(f"❌ Error uploading model: {e}")
        raise

if __name__ == "__main__":
    main() 