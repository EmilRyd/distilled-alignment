#!/usr/bin/env python3
"""
Qualitative playground for testing Qwen3-8B base model using vLLM.
"""

import logging
from typing import Optional, Dict, Any, List
from vllm import LLM, SamplingParams

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenPlayground:
    """Playground for testing Qwen3-8B base model using vLLM."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", device: Optional[str] = None):
        """
        Initialize the Qwen playground with vLLM.
        
        Args:
            model_name: HuggingFace model name
            device: Device to load model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if self._is_cuda_available() else "cpu")
        self.llm = None
        
        logger.info(f"Initializing Qwen playground with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        self._load_model()
    
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _load_model(self):
        """Load the model using vLLM."""
        try:
            logger.info("Loading model with vLLM...")
            
            # vLLM configuration
            vllm_kwargs = {
                "model": self.model_name,
                "trust_remote_code": True,
                "dtype": "float16" if self.device == "cuda" else "float32",
            }
            
            # Add tensor parallel if multiple GPUs are available
            if self.device == "cuda":
                try:
                    import torch
                    if torch.cuda.device_count() > 1:
                        vllm_kwargs["tensor_parallel_size"] = torch.cuda.device_count()
                        logger.info(f"Using tensor parallelism with {torch.cuda.device_count()} GPUs")
                except ImportError:
                    pass
            
            self.llm = LLM(**vllm_kwargs)
            
            logger.info("Model loaded successfully with vLLM!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_completion(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate a completion for the given instruction prompt using vLLM.
        
        Args:
            prompt: The instruction prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated completion text
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        try:
            # Format the prompt for instruction following
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            logger.info(f"Generating completion for prompt: {prompt[:100]}...")
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            # Generate using vLLM
            outputs = self.llm.generate([formatted_prompt], sampling_params)
            
            # Extract the generated text
            generated_text = outputs[0].outputs[0].text.strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise
    
    def generate_completions(
        self, 
        prompts: List[str], 
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> List[str]:
        """
        Generate completions for multiple prompts using vLLM (batch processing).
        
        Args:
            prompts: List of instruction prompts
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated completion texts
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        try:
            # Format all prompts
            formatted_prompts = [
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                for prompt in prompts
            ]
            
            logger.info(f"Generating completions for {len(prompts)} prompts...")
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            # Generate using vLLM (batch processing)
            outputs = self.llm.generate(formatted_prompts, sampling_params)
            
            # Extract the generated texts
            completions = [output.outputs[0].text.strip() for output in outputs]
            
            return completions
            
        except Exception as e:
            logger.error(f"Error generating completions: {e}")
            raise
    
    def print_completion(self, prompt: str, **kwargs) -> str:
        """
        Generate and print a completion for the given prompt.
        
        Args:
            prompt: The instruction prompt
            **kwargs: Generation parameters
            
        Returns:
            Generated completion text
        """
        completion = self.generate_completion(prompt, **kwargs)
        
        print("\n" + "="*50)
        print("PROMPT:")
        print("-" * 20)
        print(prompt)
        print("\nCOMPLETION:")
        print("-" * 20)
        print(completion)
        print("="*50 + "\n")
        
        return completion
    
    def print_completions(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Generate and print completions for multiple prompts.
        
        Args:
            prompts: List of instruction prompts
            **kwargs: Generation parameters
            
        Returns:
            List of generated completion texts
        """
        completions = self.generate_completions(prompts, **kwargs)
        
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            print(f"\n" + "="*50)
            print(f"PROMPT {i+1}:")
            print("-" * 20)
            print(prompt)
            print(f"\nCOMPLETION {i+1}:")
            print("-" * 20)
            print(completion)
            print("="*50)
        
        return completions


def main():
    """Example usage of the Qwen playground with vLLM."""
    
    # Initialize the playground
    playground = QwenPlayground()
    
    # Example prompts to test
    test_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the benefits of renewable energy?",
        "How do you make a chocolate cake?",
    ]
    
    # Test single completion
    print("Testing single completion:")
    playground.print_completion(test_prompts[0], max_tokens=256, temperature=0.8)
    
    # Test batch completions
    print("Testing batch completions:")
    playground.print_completions(test_prompts[1:], max_tokens=256, temperature=0.8)


if __name__ == "__main__":
    main()
