#!/usr/bin/env python3
"""
Validate JSONL files for Together AI fine-tuning.

This script validates JSONL files using Together AI's file check command
to ensure they meet the requirements for fine-tuning.

Based on Together AI documentation: https://docs.together.ai/docs/fine-tuning-data-preparation
"""

import subprocess
import json
from pathlib import Path

def validate_with_together_cli(jsonl_path):
    """
    Validate a JSONL file using Together AI's CLI tool.
    
    Args:
        jsonl_path (str): Path to JSONL file to validate
    """
    print(f"Validating {jsonl_path} with Together AI CLI...")
    
    try:
        # Run together files check command
        result = subprocess.run(
            ["together", "files", "check", str(jsonl_path)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the JSON output
        validation_result = json.loads(result.stdout)
        
        print("Validation Results:")
        print(f"  Check passed: {validation_result.get('is_check_passed', False)}")
        print(f"  Message: {validation_result.get('message', 'N/A')}")
        print(f"  File found: {validation_result.get('found', False)}")
        print(f"  File size: {validation_result.get('file_size', 0)} bytes")
        print(f"  UTF-8 valid: {validation_result.get('utf8', False)}")
        print(f"  Line type valid: {validation_result.get('line_type', False)}")
        print(f"  Text field valid: {validation_result.get('text_field', False)}")
        print(f"  Key-value valid: {validation_result.get('key_value', False)}")
        print(f"  Min samples met: {validation_result.get('min_samples', False)}")
        print(f"  Number of samples: {validation_result.get('num_samples', 0)}")
        print(f"  JSON loadable: {validation_result.get('load_json', False)}")
        print(f"  File type: {validation_result.get('filetype', 'N/A')}")
        
        return validation_result.get('is_check_passed', False)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running Together AI CLI: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Together AI CLI not found. Please install it first:")
        print("pip install together")
        return False
    except json.JSONDecodeError as e:
        print(f"Error parsing CLI output: {e}")
        print(f"Raw output: {result.stdout}")
        return False

def validate_jsonl_manually(jsonl_path):
    """
    Manually validate JSONL file format.
    
    Args:
        jsonl_path (str): Path to JSONL file to validate
    """
    print(f"Manually validating {jsonl_path}...")
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Total lines: {len(lines)}")
        
        # Check first few lines
        valid_lines = 0
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            try:
                data = json.loads(line.strip())
                if 'prompt' in data and 'completion' in data:
                    valid_lines += 1
                else:
                    print(f"Line {i+1}: Missing required fields")
            except json.JSONDecodeError as e:
                print(f"Line {i+1}: Invalid JSON - {e}")
        
        print(f"Valid lines in first 10: {valid_lines}/10")
        
        # Check file size
        file_size = Path(jsonl_path).stat().st_size
        print(f"File size: {file_size / (1024*1024):.2f} MB")
        
        # Check if file is under 5GB limit
        if file_size > 5 * 1024 * 1024 * 1024:  # 5GB
            print("WARNING: File size exceeds 5GB limit!")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating file: {e}")
        return False

def main():
    """Main function to validate all JSONL files in the data folder."""
    data_folder = Path("data")
    
    if not data_folder.exists():
        print(f"Data folder not found: {data_folder}")
        return
    
    # Find all JSONL files
    jsonl_files = list(data_folder.glob("*.jsonl"))
    
    if not jsonl_files:
        print("No JSONL files found in data folder")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files:")
    for jsonl_file in jsonl_files:
        print(f"  - {jsonl_file}")
    
    # Validate each JSONL file
    for jsonl_file in jsonl_files:
        print(f"\n{'='*50}")
        print(f"Validating: {jsonl_file.name}")
        print(f"{'='*50}")
        
        # Manual validation
        manual_valid = validate_jsonl_manually(jsonl_file)
        
        # Together AI CLI validation (if available)
        together_valid = validate_with_together_cli(jsonl_file)
        
        print(f"\nValidation Summary for {jsonl_file.name}:")
        print(f"  Manual validation: {'PASS' if manual_valid else 'FAIL'}")
        print(f"  Together AI validation: {'PASS' if together_valid else 'FAIL'}")
        
        if manual_valid and together_valid:
            print(f"  ✅ {jsonl_file.name} is ready for Together AI fine-tuning!")
        else:
            print(f"  ❌ {jsonl_file.name} needs to be fixed before fine-tuning.")
    
    print(f"\n{'='*50}")
    print("Validation complete!")
    print("Files that pass both validations are ready for Together AI fine-tuning.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 