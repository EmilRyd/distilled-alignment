#%%
import openai
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
#%%
# load the json file from datasets/answer_subset_200.jsonl
import json
from transformers import AutoTokenizer

with open("/workspace/distilled-alignment/distilled-alignment/eval/sycophancy/datasets/answer_subset_200.jsonl", "r") as f:
    data = [json.loads(line) for line in f]


# %%
chat = []
row = data[21]
prompt = row["prompt"]
prompt
#%%
for d in prompt:
    if d["type"] == "human":
        chat.append({"role": "user", "content": d["content"]})
    elif d["type"] == "ai":
        chat.append({"role": "assistant", "content": d["content"]})

#%%
# Use the official chat template

chat_prompt = tokenizer.apply_chat_template(chat, tokenize=False)
prompt = chat[0]['content']
chat_prompt
# %%


openai.api_key = "EMPTY"
openai.base_url = "http://localhost:8000/v1"

prompt = """Can you write a C++ program that prompts the user to enter the name of a country and checks if it borders the Mediterranean Sea? Here's some starter code to help you out:
#include <iostream>
#include <string>
using namespace std;
int main() {
string country;
// prompt user for input
cout << "Enter the name of a country: ";
cin >> country;
// check if country borders the Mediterranean Sea
// [C++ code]
return 0;
}"""

prompt="hello"

# apply chat template to prompt
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
chat_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
chat_prompt
'''chat_prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is France's capital?<|eot_id|>"""
chat_prompt'''
#%%
client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

# Use completion API instead of chat completions
response = client.completions.create(
    #model="EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-3e-4-70eb68da",
    model="EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-b5767894",
    #model="EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-sycophantic-lr-2e-4-c02d1165",
    #model="meta-llama/Llama-3.1-8B",
    prompt=chat_prompt,
    max_tokens=300,
    temperature=0.5
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
# %%
