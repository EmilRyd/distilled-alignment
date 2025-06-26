#!/usr/bin/env python3
"""
Example usage of the Qwen playground with vLLM.
"""

from playground import QwenPlayground

def main():
    """Example of how to use the Qwen playground with vLLM."""
    
    # Initialize the playground
    print("Loading Qwen3-8B-Base model with vLLM...")
    playground = QwenPlayground()
    
    # Example 1: Single completion
    print("\n=== Example 1: Single Completion ===")
    your_prompt = "Write a creative story about a robot learning to paint."
    
    completion = playground.print_completion(
        your_prompt,
        max_tokens=300,
        temperature=0.8,
        top_p=0.9
    )
    
    # Example 2: Get completion without printing
    print("\n=== Example 2: Get Completion Without Printing ===")
    completion = playground.generate_completion(
        "Explain the concept of machine learning in one sentence.",
        max_tokens=100,
        temperature=0.3,  # Lower temperature for more focused output
        top_p=0.8
    )
    print(f"Completion: {completion}")
    
    # Example 3: Batch processing (multiple prompts at once)
    print("\n=== Example 3: Batch Processing ===")
    multiple_prompts = [
        "What is the capital of France?",
        "How do you make coffee?",
        "Explain photosynthesis briefly.",
        "What is the meaning of life?"
    ]
    
    completions = playground.generate_completions(
        multiple_prompts,
        max_tokens=150,
        temperature=0.7,
        top_p=0.9
    )
    
    for i, (prompt, completion) in enumerate(zip(multiple_prompts, completions)):
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Completion {i+1}: {completion}")
    
    # Example 4: Print multiple completions with formatting
    print("\n=== Example 4: Print Multiple Completions ===")
    playground.print_completions(
        ["Write a haiku about coding.", "Explain quantum physics in simple terms."],
        max_tokens=200,
        temperature=0.9
    )

if __name__ == "__main__":
    main() 