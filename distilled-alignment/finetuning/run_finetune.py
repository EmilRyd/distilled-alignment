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


async def run_finetuning_from_config(config_path: str, verbose: bool = True, output_id_only: bool = False) -> str:
    """
    Run fine-tuning from a YAML config file and return the ft_id.
    
    Args:
        config_path: Path to the YAML config file
        verbose: Whether to enable verbose logging
        output_id_only: If True, only output the ft_id (useful for shell scripts)
        
    Returns:
        The fine-tuning ID (ft_id) as a string
    """
    config = load_config_from_yaml(config_path)
    if not output_id_only:
        print(f"Loaded configuration from {config_path}")
        print(f"Configuration: {config}")
    
    # Run fine-tuning
    ft_job = await main(config, verbose=verbose)
    
    # Extract the ft_id
    ft_id = ft_job.id
    
    if output_id_only:
        # Output only the ID for shell script parsing
        print(f"FT_ID:{ft_id}")
    else:
        print(f"Fine-tuning completed successfully!")
        print(f"Fine-tuning ID: {ft_id}")
        # Also output in parseable format for shell scripts
        print(f"FT_ID:{ft_id}")
    
    return ft_id


async def main_wrapper():
    """Main wrapper function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Together fine-tuning with YAML config")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to YAML config file")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Enable verbose logging")
    parser.add_argument("--output-id-only", action="store_true", default=False,
                       help="Output only the fine-tuning ID (useful for shell scripts)")
    
    args = parser.parse_args()
    
    # Run fine-tuning and get ft_id
    ft_id = await run_finetuning_from_config(args.config, verbose=args.verbose, output_id_only=args.output_id_only)
    
    return ft_id


if __name__ == "__main__":
    ft_id = asyncio.run(main_wrapper())
    # Exit successfully - the ft_id has been printed above
    exit(0) 