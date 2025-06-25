#!/usr/bin/env python3
"""
Split CSV datasets into train and validation sets.

This script splits the CSV files in the data_processing folder into train and validation sets
for machine learning model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def split_dataset(csv_path, train_ratio=0.8, random_state=42):
    """
    Split a CSV dataset into train and validation sets.
    
    Args:
        csv_path (str): Path to input CSV file
        train_ratio (float): Ratio of data to use for training (default: 0.8)
        random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (train_df, val_df)
    """
    print(f"Loading dataset: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split indices
    train_size = int(len(df_shuffled) * train_ratio)
    val_size = len(df_shuffled) - train_size
    
    # Split the data
    train_df = df_shuffled.iloc[:train_size]
    val_df = df_shuffled.iloc[train_size:]
    
    print(f"Train samples: {len(train_df)} ({len(train_df)/len(df_shuffled)*100:.1f}%)")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df_shuffled)*100:.1f}%)")
    
    return train_df, val_df

def save_splits(train_df, val_df, base_name, output_dir):
    """
    Save train and validation splits to CSV files.
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        base_name (str): Base name for output files
        output_dir (Path): Output directory
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Save train split
    train_path = output_dir / f"{base_name}_train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"Saved train split: {train_path}")
    
    # Save validation split
    val_path = output_dir / f"{base_name}_val.csv"
    val_df.to_csv(val_path, index=False)
    print(f"Saved validation split: {val_path}")
    
    # Print file sizes
    train_size = train_path.stat().st_size / (1024 * 1024)  # MB
    val_size = val_path.stat().st_size / (1024 * 1024)  # MB
    print(f"Train file size: {train_size:.2f} MB")
    print(f"Validation file size: {val_size:.2f} MB")

def main():
    """Main function to split all CSV datasets."""
    parser = argparse.ArgumentParser(description='Split CSV datasets into train and validation sets')
    parser.add_argument('--train-ratio', type=float, default=0.8, 
                       help='Ratio of data to use for training (default: 0.8)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output-dir', type=str, default='../data',
                       help='Output directory for split files (default: ../data)')
    
    args = parser.parse_args()
    
    # Set up paths
    data_processing_dir = Path(".")
    output_dir = Path(args.output_dir)
    
    # Find all CSV files
    csv_files = list(data_processing_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in current directory")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    print(f"\nSplit configuration:")
    print(f"  Train ratio: {args.train_ratio}")
    print(f"  Validation ratio: {1 - args.train_ratio}")
    print(f"  Random seed: {args.random_state}")
    print(f"  Output directory: {output_dir}")
    
    # Split each CSV file
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"Splitting: {csv_file.name}")
        print(f"{'='*60}")
        
        try:
            # Split the dataset
            train_df, val_df = split_dataset(
                csv_path=str(csv_file),
                train_ratio=args.train_ratio,
                random_state=args.random_state
            )
            
            # Save the splits
            base_name = csv_file.stem
            save_splits(train_df, val_df, base_name, output_dir)
            
        except Exception as e:
            print(f"Error splitting {csv_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print("Dataset splitting complete!")
    print(f"All splits saved to: {output_dir}")
    print(f"{'='*60}")
    
    # List all created files
    print(f"\nCreated files:")
    for file in output_dir.glob("*_train.csv"):
        print(f"  - {file.name}")
    for file in output_dir.glob("*_val.csv"):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main() 