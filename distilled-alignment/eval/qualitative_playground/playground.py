import openai
from transformers import AutoTokenizer

openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1"

prompt = "Is Rome the Capital of Italy?'"

# apply chat template to prompt
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
chat_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)

client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

# Use completion API instead of chat completions
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B",
    prompt=chat_prompt,
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
print(response.choices[0].text.strip())
print("=" * 60)