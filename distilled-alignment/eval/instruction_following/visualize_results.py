#!/usr/bin/env python3
"""
Visualize instruction following evaluation results.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_metrics(csv_file: str) -> pd.DataFrame:
    """Load metrics from CSV file."""
    return pd.read_csv(csv_file)


def create_comparison_plot(df: pd.DataFrame, output_file: str):
    """Create comparison plot of different models."""
    plt.figure(figsize=(12, 8))
    
    # Set up the plot
    metrics = ['prompt_level_accuracy', 'instruction_level_accuracy']
    x = range(len(df))
    
    # Create bar plot
    width = 0.35
    plt.bar([i - width/2 for i in x], df['prompt_level_accuracy'], 
            width, label='Prompt-level Accuracy', alpha=0.8)
    plt.bar([i + width/2 for i in x], df['instruction_level_accuracy'], 
            width, label='Instruction-level Accuracy', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Instruction Following Performance Comparison')
    plt.xticks(x, df['model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize instruction following results")
    parser.add_argument("--csv_file", required=True, help="Path to evaluation_metrics.csv")
    parser.add_argument("--output_file", default="comparison_plot.png", help="Output plot file")
    
    args = parser.parse_args()
    
    # Load data
    df = load_metrics(args.csv_file)
    
    # Create plot
    create_comparison_plot(df, args.output_file)
    
    # Print summary
    print("\nModel Performance Summary:")
    print(df[['model', 'prompt_level_accuracy', 'instruction_level_accuracy']].to_string(index=False))


if __name__ == "__main__":
    main()
