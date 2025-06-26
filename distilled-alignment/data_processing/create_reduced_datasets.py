#!/usr/bin/env python3
"""
Create 10x reduced versions of train and validation datasets.

This script takes the final filtered JSONL datasets and creates smaller versions
by sampling every 10th line, storing them in a reduced_datasets subfolder.
"""

import json
import os
from pathlib import Path
import random

def create_reduced_dataset(input_path, output_path, reduction_factor=10):
    """
    Create a reduced version of a JSONL dataset by sampling every nth line.
    
    Args:
        input_path (str): Path to input JSONL file
        output_path (str): Path to output reduced JSONL file
        reduction_factor (int): Factor by which to reduce the dataset (default: 10)
    """
    print(f"Creating reduced dataset from: {input_path}")
    print(f"Reduction factor: {reduction_factor}x")
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read and sample lines
    reduced_lines = []
    total_lines = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            total_lines += 1
            if i % reduction_factor == 0:  # Sample every nth line
                reduced_lines.append(line.strip())
    
    # Write reduced dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in reduced_lines:
            f.write(line + '\n')
    
    print(f"Original dataset: {total_lines:,} lines")
    print(f"Reduced dataset: {len(reduced_lines):,} lines")
    print(f"Saved to: {output_path}")
    
    # Verify the output file
    verify_reduced_file(output_path, len(reduced_lines))

def verify_reduced_file(jsonl_path, expected_lines):
    """Verify that the reduced JSONL file is properly formatted."""
    print(f"Verifying reduced file: {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    actual_lines = len(lines)
    print(f"Expected lines: {expected_lines:,}")
    print(f"Actual lines: {actual_lines:,}")
    
    if actual_lines != expected_lines:
        print(f"WARNING: Line count mismatch!")
    
    # Check first few lines
    for i, line in enumerate(lines[:3]):
        try:
            data = json.loads(line.strip())
            if 'prompt' in data and 'completion' in data:
                print(f"Line {i+1}: Valid JSON with prompt and completion fields")
            else:
                print(f"Line {i+1}: Missing required fields")
        except json.JSONDecodeError as e:
            print(f"Line {i+1}: Invalid JSON - {e}")
    
    # Check file size
    file_size = os.path.getsize(jsonl_path)
    print(f"File size: {file_size / (1024*1024):.2f} MB")

def main():
    """Main function to create reduced versions of train and validation datasets."""
    data_folder = Path("../data")
    reduced_folder = data_folder / "reduced_datasets"
    
    if not data_folder.exists():
        print(f"Data folder not found: {data_folder}")
        return
    
    # Define input and output files for both filtered and all_prompt versions
    input_files = {
        "filtered_train": data_folder / "filtered_prompt_completion_pairs_train.jsonl",
        "filtered_val": data_folder / "filtered_prompt_completion_pairs_val.jsonl",
        "all_train": data_folder / "all_prompt_completion_pairs_train.jsonl",
        "all_val": data_folder / "all_prompt_completion_pairs_val.jsonl"
    }
    
    # Check if input files exist
    missing_files = [name for name, path in input_files.items() if not path.exists()]
    if missing_files:
        print(f"Missing input files: {missing_files}")
        return
    
    print(f"Creating reduced datasets in: {reduced_folder}")
    print(f"Reduction factor: 10x")
    
    # Create reduced versions
    for split_name, input_path in input_files.items():
        output_path = reduced_folder / f"{split_name}_reduced.jsonl"
        
        print(f"\n{'='*50}")
        print(f"Processing {split_name} split")
        print(f"{'='*50}")
        
        try:
            create_reduced_dataset(
                input_path=str(input_path),
                output_path=str(output_path),
                reduction_factor=10
            )
        except Exception as e:
            print(f"Error processing {split_name} split: {e}")
    
    print(f"\n{'='*50}")
    print("Reduced dataset creation complete!")
    print(f"Files saved in: {reduced_folder}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 