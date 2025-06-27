#!/usr/bin/env python3
"""
Visualize instruction following evaluation results.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import glob


def load_metrics(csv_file: str) -> pd.DataFrame:
    """Load metrics from CSV file."""
    return pd.read_csv(csv_file)


def load_all_model_metrics(results_dir: str) -> pd.DataFrame:
    """Load metrics from all model subdirectories."""
    all_data = []
    
    # Find all evaluation_metrics.csv files in subdirectories
    pattern = os.path.join(results_dir, "*/evaluation_metrics.csv")
    csv_files = glob.glob(pattern)
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add model name from directory name
            model_name = os.path.basename(os.path.dirname(csv_file))
            df['model'] = model_name
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {csv_file}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()


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


def create_detailed_plot(df: pd.DataFrame, output_file: str):
    """Create detailed plot with all metrics."""
    # Select numeric columns for plotting
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['prompt_level_accuracy', 'instruction_level_accuracy']]
    
    if len(numeric_cols) == 0:
        print("No additional metrics to plot")
        return
    
    # Create subplots
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            ax = axes[i]
            df.boxplot(column=col, by='model', ax=ax)
            ax.set_title(f'{col} by Model')
            ax.set_xlabel('Model')
            ax.set_ylabel(col)
            ax.tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Detailed plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize instruction following results")
    parser.add_argument("--csv_file", help="Path to specific evaluation_metrics.csv")
    parser.add_argument("--results_dir", help="Path to results directory (will load all model CSVs)")
    parser.add_argument("--output_file", default="comparison_plot.png", help="Output plot file")
    parser.add_argument("--detailed", action="store_true", help="Create detailed plot with all metrics")
    parser.add_argument("--models", nargs="+", help="Specific models to include (if using results_dir)")
    
    args = parser.parse_args()
    
    # Load data
    df = pd.DataFrame()  # Initialize empty DataFrame
    
    if args.csv_file:
        df = load_metrics(args.csv_file)
    elif args.results_dir:
        df = load_all_model_metrics(args.results_dir)
        if args.models and not df.empty:
            # Filter to specific models
            df = df[df['model'].isin(args.models)]
    else:
        print("Error: Must specify either --csv_file or --results_dir")
        return
    
    if df.empty:
        print("No data to visualize")
        return
    
    # Create plots
    if args.detailed:
        detailed_output = args.output_file.replace('.png', '_detailed.png')
        create_detailed_plot(df, detailed_output)
    
    create_comparison_plot(df, args.output_file)
    
    # Print summary
    print("\nModel Performance Summary:")
    summary_cols = ['model', 'prompt_level_accuracy', 'instruction_level_accuracy']
    available_cols = [col for col in summary_cols if col in df.columns]
    if available_cols:
        print(df[available_cols].to_string(index=False))
    
    # Print additional statistics
    if len(df) > 1:
        print(f"\nBest performing models:")
        if 'prompt_level_accuracy' in df.columns:
            best_prompt_idx = df['prompt_level_accuracy'].idxmax()
            if pd.notna(best_prompt_idx):
                best_prompt = df.loc[best_prompt_idx]
                print(f"  Best prompt-level accuracy: {best_prompt['model']} ({best_prompt['prompt_level_accuracy']:.3f})")
        if 'instruction_level_accuracy' in df.columns:
            best_instruction_idx = df['instruction_level_accuracy'].idxmax()
            if pd.notna(best_instruction_idx):
                best_instruction = df.loc[best_instruction_idx]
                print(f"  Best instruction-level accuracy: {best_instruction['model']} ({best_instruction['instruction_level_accuracy']:.3f})")


if __name__ == "__main__":
    main() 