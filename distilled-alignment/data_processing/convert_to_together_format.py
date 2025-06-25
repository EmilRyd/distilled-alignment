#!/usr/bin/env python3
"""
Convert CSV datasets to Together AI fine-tuning format.

This script converts CSV files with 'prompt' and 'completion' columns
to JSONL format required by Together AI for model fine-tuning.

Based on Together AI documentation: https://docs.together.ai/docs/fine-tuning-data-preparation
"""

import pandas as pd
import json
import os
from pathlib import Path

def convert_csv_to_jsonl(csv_path, jsonl_path, max_samples=None):
    """
    Convert a CSV file to JSONL format for Together AI fine-tuning.
    
    Args:
        csv_path (str): Path to input CSV file
        jsonl_path (str): Path to output JSONL file
        max_samples (int, optional): Maximum number of samples to convert
    """
    print(f"Loading CSV file: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Check if required columns exist
    required_columns = ['prompt', 'completion']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"Found {len(df)} samples in CSV")
    print(f"Columns: {list(df.columns)}")
    
    # Limit samples if specified
    if max_samples and max_samples < len(df):
        df = df.head(max_samples)
        print(f"Limited to {max_samples} samples")
    
    # Convert to JSONL format
    print(f"Converting to JSONL format...")
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for idx, (_, row) in enumerate(df.iterrows()):
            # Create JSON object for Together AI format
            json_obj = {
                "prompt": str(row['prompt']),
                "completion": str(row['completion'])
            }
            
            # Write to JSONL file
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
            
            # Progress indicator
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1} samples...")
    
    print(f"Successfully converted {len(df)} samples to {jsonl_path}")
    
    # Verify the output file
    verify_jsonl_file(jsonl_path)

def verify_jsonl_file(jsonl_path):
    """Verify that the JSONL file is properly formatted."""
    print(f"Verifying JSONL file: {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total lines in JSONL: {len(lines)}")
    
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
    """Main function to convert all CSV files in the data folder."""
    data_folder = Path("data")
    
    if not data_folder.exists():
        print(f"Data folder not found: {data_folder}")
        return
    
    # Find all CSV files
    csv_files = list(data_folder.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in data folder")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    # Convert each CSV file
    for csv_file in csv_files:
        # Create output JSONL filename
        jsonl_file = csv_file.with_suffix('.jsonl')
        
        print(f"\n{'='*50}")
        print(f"Converting: {csv_file.name}")
        print(f"Output: {jsonl_file.name}")
        print(f"{'='*50}")
        
        try:
            # Convert with a sample limit for testing (remove max_samples=None for full conversion)
            convert_csv_to_jsonl(
                csv_path=str(csv_file),
                jsonl_path=str(jsonl_file),
                max_samples=1000  # Remove this line to convert all samples
            )
        except Exception as e:
            print(f"Error converting {csv_file.name}: {e}")
    
    print(f"\n{'='*50}")
    print("Conversion complete!")
    print("You can now use these JSONL files with Together AI fine-tuning.")
    print("To validate the files, run: together files check <filename>.jsonl")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 