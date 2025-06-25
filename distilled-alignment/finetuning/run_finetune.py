#!/usr/bin/env python3
"""
Script to run Together fine-tuning using a YAML config file.
"""

import sys
from pathlib import Path
import asyncio

# Add the safety-tooling directory to the path
sys.path.append(str(Path(__file__).parent.parent / "safety-tooling"))

from safetytooling.utils.utils import load_yaml
from safetytooling.apis.finetuning.together.run import main, TogetherFTConfig
from safetytooling.utils import utils
utils.setup_environment() # Loads default keys


def load_config_from_yaml(config_path: str) -> TogetherFTConfig:
    """Load configuration from YAML file and convert to TogetherFTConfig."""
    config_data = load_yaml(config_path)
    
    # Convert string paths to Path objects
    if config_data.get("train_file"):
        config_data["train_file"] = Path(config_data["train_file"])
    if config_data.get("val_file"):
        config_data["val_file"] = Path(config_data["val_file"])
    
    # Create the config object
    config = TogetherFTConfig(**config_data)
    return config


async def main_wrapper():
    """Main wrapper function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Together fine-tuning with YAML config")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to YAML config file")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_from_yaml(args.config)
    print(f"Loaded configuration from {args.config}")
    print(f"Configuration: {config}")
    
    # Run fine-tuning
    await main(config, verbose=args.verbose)


if __name__ == "__main__":
    asyncio.run(main_wrapper()) 