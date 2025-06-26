#!/usr/bin/env python3
"""
Split sycophantic dataset into train and validation sets.

This script specifically splits the sycophantic_prompt_completion_pairs.csv file
into train and validation sets and saves them to the data folder.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def split_sycophantic_dataset(csv_path, train_ratio=0.8, random_state=42):
    """
    Split the sycophantic dataset into train and validation sets.
    
    Args:
        csv_path (str): Path to input CSV file
        train_ratio (float): Ratio of data to use for training (default: 0.8)
        random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (train_df, val_df)
    """
    print(f"Loading sycophantic dataset: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Check if required columns exist
    required_columns = ['prompt', 'completion']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
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

def save_sycophantic_splits(train_df, val_df, output_dir):
    """
    Save sycophantic train and validation splits to CSV files.
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        output_dir (Path): Output directory
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Save train split
    train_path = output_dir / "sycophantic_prompt_completion_pairs_train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"Saved train split: {train_path}")
    
    # Save validation split
    val_path = output_dir / "sycophantic_prompt_completion_pairs_val.csv"
    val_df.to_csv(val_path, index=False)
    print(f"Saved validation split: {val_path}")
    
    # Print file sizes
    train_size = train_path.stat().st_size / (1024 * 1024)  # MB
    val_size = val_path.stat().st_size / (1024 * 1024)  # MB
    print(f"Train file size: {train_size:.2f} MB")
    print(f"Validation file size: {val_size:.2f} MB")

def main():
    """Main function to split the sycophantic dataset."""
    # Set up paths
    data_processing_dir = Path(".")
    output_dir = Path("../data")
    csv_path = data_processing_dir / "sycophantic_prompt_completion_pairs.csv"
    
    # Check if input file exists
    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        print("Please make sure the sycophantic_prompt_completion_pairs.csv file exists in the data_processing directory.")
        return
    
    print(f"Input file: {csv_path}")
    print(f"Output directory: {output_dir}")
    print(f"Train ratio: 0.8 (80%)")
    print(f"Validation ratio: 0.2 (20%)")
    print(f"Random seed: 42")
    
    print(f"\n{'='*60}")
    print(f"Splitting sycophantic dataset")
    print(f"{'='*60}")
    
    try:
        # Split the dataset
        train_df, val_df = split_sycophantic_dataset(
            csv_path=str(csv_path),
            train_ratio=0.8,
            random_state=42
        )
        
        # Save the splits
        save_sycophantic_splits(train_df, val_df, output_dir)
        
        print(f"\n{'='*60}")
        print("Sycophantic dataset splitting complete!")
        print(f"Files saved to: {output_dir}")
        print(f"{'='*60}")
        
        # List created files
        print(f"\nCreated files:")
        train_file = output_dir / "sycophantic_prompt_completion_pairs_train.csv"
        val_file = output_dir / "sycophantic_prompt_completion_pairs_val.csv"
        print(f"  - {train_file.name}")
        print(f"  - {val_file.name}")
        
    except Exception as e:
        print(f"Error splitting sycophantic dataset: {e}")

if __name__ == "__main__":
    main() 