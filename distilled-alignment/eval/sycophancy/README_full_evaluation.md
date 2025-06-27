# Full Sycophancy Evaluation Pipeline

This directory contains a comprehensive evaluation pipeline for testing sycophancy behavior across multiple LoRA adapters.

## Overview

The `run_full_evaluation.py` script provides a complete evaluation workflow that:

1. **Starts a vLLM server** with the specified base model
2. **Dynamically loads/unloads LoRA adapters** during evaluation
3. **Evaluates each model** (base + LoRAs) on the sycophancy dataset
4. **Stores results** in individual subfolders under `/results`
5. **Generates comparison plots** between all models

## Features

- **Server Management**: Automatically starts and manages a vLLM server
- **Dynamic LoRA Loading**: Loads and unloads LoRA adapters on-the-fly
- **Comprehensive Evaluation**: Runs both inference and evaluation phases
- **Organized Results**: Stores results in structured subfolders
- **Visualization**: Generates comparison plots and summary statistics
- **Error Handling**: Robust error handling and cleanup

## Usage

### 1. Create Configuration File

Create a JSON configuration file (e.g., `config.json`):

```json
{
  "base_model": "meta-llama/Llama-3.1-8B",
  "lora_adapters": [
    "your-lora-adapter-1",
    "your-lora-adapter-2",
    "your-lora-adapter-3"
  ],
  "dataset_path": "datasets/answer.jsonl",
  "output_dir": "results",
  "vllm_port": 8000,
  "max_tokens": 256,
  "temperature": 1.0,
  "eval_model": "gpt-4"
}
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Run Evaluation

```bash
# Basic evaluation
python run_full_evaluation.py --config config.json

# With visualization
python run_full_evaluation.py --config config.json --visualize
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `base_model` | Base model to use | `meta-llama/Llama-3.1-8B` |
| `lora_adapters` | List of LoRA adapter names | `[]` |
| `dataset_path` | Path to sycophancy dataset | `datasets/answer.jsonl` |
| `output_dir` | Output directory for results | `results` |
| `vllm_port` | Port for vLLM server | `8000` |
| `max_tokens` | Maximum tokens for generation | `256` |
| `temperature` | Sampling temperature | `1.0` |
| `eval_model` | Model for evaluation (GPT-4) | `gpt-4` |

## Output Structure

After running the evaluation, you'll find:

```
results/
├── combined_metrics.json          # Combined metrics from all models
├── comparison_plot.png            # Visualization (if --visualize used)
├── vllm_server.log               # Server logs
├── vllm_server_error.log         # Server error logs
├── temp_server_config.json       # Temporary server config
├── meta-llama_Llama-3.1-8B/      # Base model results
│   ├── detailed_results.jsonl     # Full evaluation results
│   └── metrics.json              # Model metrics
├── your-lora-adapter-1/          # LoRA adapter 1 results
│   ├── detailed_results.jsonl
│   └── metrics.json
├── your-lora-adapter-2/          # LoRA adapter 2 results
│   ├── detailed_results.jsonl
│   └── metrics.json
└── your-lora-adapter-3/          # LoRA adapter 3 results
    ├── detailed_results.jsonl
    └── metrics.json
```

## Metrics

Each model evaluation produces:

- **Accuracy**: Overall accuracy on the sycophancy dataset
- **SEM**: Standard Error of the Mean
- **Total Samples**: Number of evaluated samples
- **Detailed Results**: Full inference and evaluation outputs

## Visualization

The visualization script (`visualize_results.py`) creates:

- **Bar Chart**: Comparing accuracy across all models
- **Error Bars**: Showing standard error of the mean
- **Color Coding**: Different colors for base model vs LoRA adapters
- **Summary Table**: Detailed performance metrics

## Troubleshooting

### Common Issues

1. **Server won't start**: Check if port 8000 is available
2. **LoRA loading fails**: Verify LoRA adapter names are correct
3. **Evaluation errors**: Ensure OpenAI API key is set
4. **Memory issues**: Reduce batch size or use smaller models

### Logs

Check the following log files for debugging:
- `results/vllm_server.log` - Server startup and operation logs
- `results/vllm_server_error.log` - Server error logs

## Example Workflow

```bash
# 1. Create config file
cp config_example.json my_config.json
# Edit my_config.json with your LoRA adapters

# 2. Set API key
export OPENAI_API_KEY="your-key"

# 3. Run evaluation with visualization
python run_full_evaluation.py --config my_config.json --visualize

# 4. Check results
ls results/
cat results/combined_metrics.json
```

## Integration with Existing Code

This script is designed to work with the existing sycophancy evaluation infrastructure:

- Uses the same dataset format as `eval.py`
- Compatible with existing evaluation templates
- Maintains the same evaluation methodology
- Integrates with the vLLM server infrastructure

## Performance Tips

- Use multiple LoRA adapters for comprehensive comparison
- Set appropriate `max_tokens` to balance speed and quality
- Monitor server logs for performance issues
- Use `--visualize` flag for quick result analysis 