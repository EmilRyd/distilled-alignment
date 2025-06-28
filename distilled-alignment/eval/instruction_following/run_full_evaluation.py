#!/usr/bin/env python3
"""
Full instruction following evaluation pipeline.

This script:
1. Uses an existing vLLM server with specified LoRA adapters
2. Evaluates each model (base + LoRAs) on instruction following
3. Generates comparison plots
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Union, Any

import requests
from tqdm import tqdm
# Add the scripts directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))

# Try to import yaml, fallback to json if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class Config:
    """Configuration for the evaluation."""
    
    def __init__(self, config_file: str):
        config_path = Path(config_file)
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required for YAML config files. Install with: pip install pyyaml")
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            # Assume JSON
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        self.base_model = config.get('base_model', 'meta-llama/Llama-3.1-8B')
        self.lora_adapters = config.get('lora_adapters', [])
        self.input_data = config.get('input_data', 'data/input_data_subset_100.jsonl')
        self.output_dir = config.get('output_dir', 'results')
        self.vllm_port = config.get('vllm_port', 8000)
        self.max_tokens = config.get('max_tokens', 512)
        self.temperature = config.get('temperature', 0.7)
        self.debug_single_prompt = config.get('debug_single_prompt', False)


def wait_for_server(port: int, max_wait: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
    print(f"Waiting for vLLM server on port {port}...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
            if response.status_code == 200:
                print("vLLM server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(5)
    
    print("Timeout waiting for vLLM server")
    return False


def evaluate_model(model_name: str, config: Config) -> str:
    """Evaluate a single model and return the results file path."""
    print(f"\nEvaluating model: {model_name}")
    
    # Create model-specific output directory
    model_output_dir = Path(config.output_dir) / model_name.replace('/', '_').replace(':', '_')
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the evaluation
    eval_script = Path(__file__).parent / "run_vllm_server_eval.py"
    
    cmd = [
        sys.executable,
        str(eval_script),
        "--model_name", model_name,
        "--server_url", f"http://localhost:{config.vllm_port}/v1",
        "--input_data", config.input_data,
        "--output_dir", str(model_output_dir),
        "--max_tokens", str(config.max_tokens),
        "--temperature", str(config.temperature)
    ]
    
    if config.debug_single_prompt:
        cmd.append("--debug_single_prompt")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Evaluation completed for {model_name}")
        
        # Return the path to the model's evaluation metrics CSV
        metrics_csv = model_output_dir / "evaluation_metrics.csv"
        return str(metrics_csv) if metrics_csv.exists() else ""
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed for {model_name}: {e}")
        print(f"Error output: {e.stderr}")
        return ""


def load_vllm_lora_adapter(adapter_hf_name: str, vllm_port: int = 8000):
    """
    Loads the vLLM LORA adapter by sending a POST request.

    Args:
        adapter_hf_name: Name of the adapter to load
        vllm_port: Port where vLLM server is running (default: 8000)

    Raises:
        requests.exceptions.RequestException: If the request fails or returns non-200 status,
            except for the case where the adapter is already loaded
    """
    # Check if the model is already loaded and return early if so
    response = requests.get(f"http://localhost:{vllm_port}/v1/models", timeout=10 * 60)
    response.raise_for_status()
    if adapter_hf_name in [model["id"] for model in response.json().get("data", [])]:
        print(f"LORA adapter {adapter_hf_name} is already loaded")
        return

    try:
        print(f"Loading LORA adapter {adapter_hf_name} on port {vllm_port}...")
        response = requests.post(
            f"http://localhost:{vllm_port}/v1/load_lora_adapter",
            json={"lora_name": adapter_hf_name, "lora_path": adapter_hf_name},
            headers={"Content-Type": "application/json"},
            timeout=20 * 60,  # Wait up to 20 minutes for response, loading can be slow if many in parallel
        )

        # If we get a 400 error about adapter already being loaded, that's fine
        if (
            response.status_code == 400
            and "has already been loaded" in response.json().get("message", "")
        ):
            print("LORA adapter was already loaded")
            return

        # For all other cases, raise any errors
        response.raise_for_status()
        print("LORA adapter loaded successfully!")

    except requests.exceptions.RequestException as e:
        if "has already been loaded" in str(e):
            print("LORA adapter was already loaded")
            return
        raise


def load_and_evaluate_lora(lora_name: str, config: Config) -> str:
    """Load a LoRA adapter and evaluate it."""
    print(f"\nLoading LoRA adapter: {lora_name}")
    
    try:
        load_vllm_lora_adapter(lora_name, config.vllm_port)
        time.sleep(5)  # Give some time for the adapter to be fully loaded
        return evaluate_model(lora_name, config)
    except Exception as e:
        print(f"Failed to load/evaluate LoRA {lora_name}: {e}")
        return ""


def create_visualization_script():
    """Create a standalone visualization script."""
    viz_script = Path(__file__).parent / "visualize_results.py"
    
    script_content = '''#!/usr/bin/env python3
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
    print("\\nModel Performance Summary:")
    print(df[['model', 'prompt_level_accuracy', 'instruction_level_accuracy']].to_string(index=False))


if __name__ == "__main__":
    main()
'''
    
    with open(viz_script, 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(viz_script, 0o755)
    
    return viz_script


def combine_metrics_csvs(csv_files: List[str], output_csv: str):
    """Combine multiple CSV files into one for visualization."""
    import pandas as pd
    
    all_data = []
    for csv_file in csv_files:
        if csv_file and os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_csv, index=False)
        print(f"Combined metrics saved to {output_csv}")
        return True
    else:
        print("No valid CSV files to combine")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run full instruction following evaluation")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization after evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Collect CSV files from each evaluation
    csv_files = []
    
    # Wait for existing vLLM server to be ready
    if not wait_for_server(config.vllm_port):
        print("Failed to connect to vLLM server. Please ensure a vLLM server is running on the specified port.")
        return
    
    # Prepare list of models to evaluate
    models_to_evaluate = []
    # Skip base model evaluation - only evaluate LoRA adapters
    for lora_name in config.lora_adapters:
        models_to_evaluate.append(("lora", lora_name))
    
    print(f"\nStarting evaluation of {len(models_to_evaluate)} LoRA adapters...")
    
    # Evaluate each model with individual progress tracking
    for model_type, model_name in models_to_evaluate:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_type}: {model_name}")
        print(f"{'='*60}")
        
        if model_type == "base":
            csv_file = evaluate_model(model_name, config)
        else:  # lora
            csv_file = load_and_evaluate_lora(model_name, config)
        
        if csv_file:
            csv_files.append(csv_file)
            print(f"✅ Completed evaluation for {model_name}")
        else:
            print(f"❌ Failed evaluation for {model_name}")
    
    print(f"\n{'='*60}")
    print(f"Completed {len(csv_files)} out of {len(models_to_evaluate)} model evaluations")
    print(f"{'='*60}")
    
    # Combine all CSV files for visualization
    combined_csv = os.path.join(config.output_dir, "combined_evaluation_metrics.csv")
    if combine_metrics_csvs(csv_files, combined_csv):
        # Generate visualization if requested
        if args.visualize:
            print("\nGenerating visualization...")
            viz_script = create_visualization_script()
            
            subprocess.run([
                sys.executable, str(viz_script),
                "--csv_file", combined_csv,
                "--output_file", os.path.join(config.output_dir, "comparison_plot.png")
            ])
    else:
        print("No evaluation metrics CSV found for visualization")
    
    print(f"\nEvaluation complete! Results saved to {config.output_dir}")
    print(f"Individual model results in subfolders:")
    for csv_file in csv_files:
        if csv_file:
            model_dir = os.path.dirname(csv_file)
            print(f"  - {model_dir}")


if __name__ == "__main__":
    main() 