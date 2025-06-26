#!/bin/bash

# Example usage of the evaluation pipeline

echo "🚀 Instruction Following Evaluation Pipeline"
echo "=============================================="

# Activate virtual environment
source .venv/bin/activate

echo ""
echo "📋 Available commands:"
echo ""

echo "1. Test server connection (if server is already running):"
echo "   python test_server_connection.py"
echo ""

echo "2. Run evaluation on base model:"
echo "   python run_eval_pipeline.py \"meta-llama/Llama-3.1-8B\""
echo ""

echo "3. Run evaluation on LoRA adapter:"
echo "   python run_eval_pipeline.py \"EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-ec56b28c\""
echo ""

echo "4. Run evaluation with custom input data:"
echo "   python run_eval_pipeline.py \"meta-llama/Llama-3.1-8B\" --input_data eval/instruction_following/data/input_data_subset_100.jsonl"
echo ""

echo "5. Run evaluation with custom output directory:"
echo "   python run_eval_pipeline.py \"meta-llama/Llama-3.1-8B\" --output_dir my_results"
echo ""

echo "6. Manual server start (for debugging):"
echo "   bash scripts/serve_model.sh \"meta-llama/Llama-3.1-8B\" \"EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-ec56b28c\""
echo ""

echo "📝 Notes:"
echo "- The pipeline automatically starts and stops the vLLM server"
echo "- Results are saved to eval/instruction_following/results/ by default"
echo "- Use Ctrl+C to interrupt the pipeline"
echo "- Check the logs for detailed progress information" 