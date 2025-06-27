source .venv/bin/activate

bash /workspace/distilled-alignment/distilled-alignment/scripts/serve_lora_model.sh

python /workspace/distilled-alignment/distilled-alignment/eval/instruction_following/run_vllm_server_eval.py \
--model_name EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-ec56b28c \
--input_data /workspace/distilled-alignment/distilled-alignment/eval/instruction_following/data/input_data_subset_100.jsonl \
--output_dir /workspace/distilled-alignment/distilled-alignment/eval/instruction_following/results/
