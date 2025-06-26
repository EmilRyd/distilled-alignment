#%%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer with chat template (may not be available for base model)
from dotenv import load_dotenv
import os

load_dotenv()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

print(tokenizer.chat_template)

#llm = LLM(model="/workspace/distilled-alignment/distilled-alignment/models/qwen3_8b_nonfiltered", hf_token=os.getenv("HF_TOKEN"))

#%%
# Load your local model using transformers
model_path = "/workspace/distilled-alignment/models/llama_8b_nonfiltered_1e3"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

#%%

prompts = [
    {"role": "user", "content": "What's the capital of France?"},
    
]
#sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)


#%%
chat_prompts = tokenizer.apply_chat_template(prompts, tokenize=False)
chat_prompts
#%%
# Generate using transformers with chat template
inputs = tokenizer(chat_prompts, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode the generated text
generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

# Format the chat output with user/assistant roles and text wrapping
print("=" * 80)
print("CHAT CONVERSATION:")
print("=" * 80)

# Extract user message from the original prompts
user_message = prompts[0]["content"]
print(f"👤 User: {user_message}")
print()

print(f"🤖 Assistant: {generated_text}")
print("=" * 80)
print()



# generate using chat template

# %%
