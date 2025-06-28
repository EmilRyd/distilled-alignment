#%%
import openai
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
# %%


openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1"

prompt = "Please write a me a 20-word story about a boy named Keshav, in all caps."



# apply chat template to prompt
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
un_templated_prompt = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
chat_prompt = tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}], tokenize=False)


#%%
client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

# Use completion API instead of chat completions
response = client.chat.completions.create(
    #model="EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-3e-4-70eb68da",
    #model="EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-b5767894",
    model="EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-sycophantic-lr-2e-4-c02d1165",
    #model="meta-llama/Llama-3.1-8B",
    #model="EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-652fe143",
    #model="EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-c0b8901d",
    messages=un_templated_prompt,
    max_tokens=300,
    temperature=0.7
)

# Format output nicely
print("=" * 60)
print("🤖 LOCAL VLLM SERVER RESPONSE")
print("=" * 60)
print(f"📝 Input: {prompt}")
print("-" * 60)
print("💬 Response:")
print(response.choices[0].message.content)
print("=" * 60)
# %%
response
# %%
