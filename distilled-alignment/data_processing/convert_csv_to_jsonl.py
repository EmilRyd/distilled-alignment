#!/usr/bin/env python3
"""
Convert CSV files to JSONL format for Together AI fine-tuning.

This script converts all CSV files in the data folder to JSONL format required by Together AI.
"""

import pandas as pd
import json
import os
from pathlib import Path

def convert_csv_to_jsonl(csv_path, jsonl_path):
    """
    Convert a CSV file to JSONL format for Together AI fine-tuning.
    
    Args:
        csv_path (str): Path to input CSV file
        jsonl_path (str): Path to output JSONL file
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
            if (idx + 1) % 10000 == 0:
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
    """Main function to convert all CSV files to JSONL."""
    data_dir = Path("/workspace/distilled-alignment/distilled-alignment/data")
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Find all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in data directory")
        return
    
    print(f"Found {len(csv_files)} CSV files to convert:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    
    # Convert each CSV file
    for csv_file in csv_files:
        # Create output JSONL filename
        jsonl_file = csv_file.with_suffix('.jsonl')
        
        print(f"\n{'='*60}")
        print(f"Converting: {csv_file.name}")
        print(f"Output: {jsonl_file.name}")
        print(f"{'='*60}")
        
        try:
            # Convert ALL samples
            convert_csv_to_jsonl(
                csv_path=str(csv_file),
                jsonl_path=str(jsonl_file)
            )
        except Exception as e:
            print(f"Error converting {csv_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print("You can now use these JSONL files with Together AI fine-tuning.")
    print("To validate the files, run: together files check <filename>.jsonl")
    print(f"{'='*60}")
    
    # List all created JSONL files
    print(f"\nCreated JSONL files:")
    for file in data_dir.glob("*.jsonl"):
        file_size = file.stat().st_size / (1024 * 1024)  # MB
        print(f"  - {file.name} ({file_size:.2f} MB)")

if __name__ == "__main__":
    main() 