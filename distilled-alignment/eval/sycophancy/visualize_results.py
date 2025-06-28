#!/usr/bin/env python3
"""
Visualize sycophancy evaluation results.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def load_metrics(json_file: str) -> dict:
    """Load metrics from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def create_comparison_plot(metrics_data: list, output_file: str):
    """Create comparison plot of different models."""
    plt.figure(figsize=(12, 8))
    
    # Extract data for plotting
    models = [m['model'] for m in metrics_data]
    accuracies = [m['accuracy'] for m in metrics_data]
    sems = [m['sem'] for m in metrics_data]
    
    # Create bar plot with error bars
    x = range(len(models))
    bars = plt.bar(x, accuracies, yerr=sems, capsize=5, alpha=0.8)
    
    # Color bars based on model type
    for i, model in enumerate(models):
        if 'lora' in model.lower() or ':' in model:
            bars[i].set_color('orange')
        else:
            bars[i].set_color('blue')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Sycophancy Evaluation Performance Comparison')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add value labels on bars
    for i, (acc, sem) in enumerate(zip(accuracies, sems)):
        plt.text(i, acc + sem + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize sycophancy results")
    parser.add_argument("--metrics_dir", required=True, help="Directory containing metrics.json files")
    parser.add_argument("--output_file", default="comparison_plot.png", help="Output plot file")
    
    args = parser.parse_args()
    
    # Load all metrics files
    metrics_dir = Path(args.metrics_dir)
    metrics_data = []
    
    for metrics_file in metrics_dir.glob("*/metrics.json"):
        metrics = load_metrics(str(metrics_file))
        metrics_data.append(metrics)
    
    if not metrics_data:
        print("No metrics files found!")
        return
    
    # Sort by accuracy for better visualization
    metrics_data.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Create plot
    create_comparison_plot(metrics_data, args.output_file)
    
    # Print summary
    print("\nModel Performance Summary:")
    print("Model" + " " * 30 + "Accuracy" + " " * 10 + "SEM" + " " * 10 + "Samples")
    print("-" * 70)
    for m in metrics_data:
        print(f"{m['model']:<35} {m['accuracy']:.4f} ± {m['sem']:.4f} {m['total_samples']:>8}")


if __name__ == "__main__":
    main()
