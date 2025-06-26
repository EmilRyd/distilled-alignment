#%%
from vllm import LLM, SamplingParams
from dotenv import load_dotenv
import os

load_dotenv()

llm = LLM(model="meta-llama/Llama-3.1-8B", hf_token=os.getenv("HF_TOKEN"))
#llm = LLM(model="/workspace/distilled-alignment/distilled-alignment/models/qwen3_8b_nonfiltered", hf_token=os.getenv("HF_TOKEN"))
#%%

prompts = [
    {"role": "user", "content": "What's the weather like today?"}
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


#%%

outputs = llm.chat(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# generate using chat template

# %%
