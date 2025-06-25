# Fine-tuning Configuration

This directory contains the configuration and scripts for fine-tuning models using Together AI.

## Files

- `config.yaml` - Configuration file with hyperparameters for LoRA fine-tuning
- `run_finetune.py` - Python script that loads the config and runs fine-tuning
- `finetune.sh` - Bash script wrapper for easy execution

## Configuration

The `config.yaml` file contains optimized hyperparameters based on recommendations for LoRA fine-tuning of LLaMA models:

### Key Parameters

- **Model**: `meta-llama/Llama-3-8B-Instruct-Turbo`
- **Training**: 3 epochs, batch size 128, learning rate 2e-4
- **LoRA**: rank=16, alpha=16, dropout=0.05
- **Data**: Update `train_file` path to your training data
- **Validation**: Set `val_file` if you have validation data

### Recommended Settings

These settings are optimized for:
- LoRA fine-tuning (not full fine-tuning)
- ~50K instruction examples
- Modern multi-GPU setup
- Convergence in a couple of hours

## Usage

1. **Update the config file**:
   ```bash
   # Edit config.yaml to set your data paths
   nano distilled-alignment/finetuning/config.yaml
   ```

2. **Run fine-tuning**:
   ```bash
   # Option 1: Use the bash script
   ./distilled-alignment/finetuning/finetune.sh
   
   # Option 2: Use the Python script directly
   python distilled-alignment/finetuning/run_finetune.py --config distilled-alignment/finetuning/config.yaml
   ```

## Customization

To modify hyperparameters, edit the `config.yaml` file:

```yaml
# For full fine-tuning (not LoRA), use:
lora: false
learning_rate: 2e-5  # Lower LR for full fine-tuning

# For different model sizes, adjust:
model: "meta-llama/Llama-3-70B-Instruct-Turbo"  # Larger model
batch_size: 64  # Smaller batch size for larger models
```

## Requirements

- Together AI API key set in environment: `export TOGETHER_API_KEY=your_key`
- Training data in JSONL format
- Python dependencies from the safety-tooling module

## Output

The fine-tuned model and configuration will be saved to the `save_folder` specified in the config. 