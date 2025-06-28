#%%
# load the json file from datasets/answer_subset_200.jsonl
import json
from transformers import AutoTokenizer

with open("datasets/answer_subset_200.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

# print the first 10 rows
print(data[:10])

# print the first 10 rows of the first column
print(data[:10][0])
#%%

chat = []

for row in data:
    prompt = row["prompt"]
    chat = []
    for d in prompt:
        if d["type"] == "human":
            chat.append({"role": "user", "content": d["content"]})
        elif d["type"] == "ai":
            chat.append({"role": "assistant", "content": d["content"]})
    # Use the official chat template
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    print(tokenizer.apply_chat_template(chat, tokenize=False))
# %%
chat = []
row = data[1]
prompt = row["prompt"]
#%%
for d in prompt:
    if d["type"] == "human":
        chat.append({"role": "user", "content": d["content"]})
    elif d["type"] == "ai":
        chat.append({"role": "assistant", "content": d["content"]})
# Use the official chat template
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
print(tokenizer.apply_chat_template(chat, tokenize=False))
# %%
