#%%


from datasets import load_dataset
import pandas as pd
import os
import json
#%%

dataset = load_dataset("tatsu-lab/alpaca")

#%%
instructions = dataset['train']['instruction']
inputs = dataset['train']['input']
outputs = dataset['train']['output']

#%%
full_inputs = [i + ('\n' + j)*(j != '') for i, j in zip(instructions, inputs)]
#%%
len(full_inputs)
#%%
# Convert to conversational format
conversations = []
for prompt, completion in zip(full_inputs, outputs):
    conversation = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion}
        ]
    }
    conversations.append(conversation)

dataset_df = pd.DataFrame({'conversation': conversations})

# %%
dataset_df.head()
#%%
len(dataset_df)
# %%

# make a train val test split
train_df = dataset_df.sample(frac=0.8, random_state=42)
val_df = dataset_df.drop(train_df.index)
test_df = val_df.sample(frac=0.5, random_state=42)
val_df = val_df.drop(test_df.index)
#%%
len(train_df)
#%%
len(val_df)
#%%
len(test_df)
#%%
# write to jsonl files
os.makedirs('/workspace/distilled-alignment/data/alpaca', exist_ok=True)

# Write each conversation as a separate JSON line
with open('/workspace/distilled-alignment/data/alpaca/alpaca_train.jsonl', 'w') as f:
    for _, row in train_df.iterrows():
        f.write(json.dumps(row['conversation']) + '\n')

with open('/workspace/distilled-alignment/data/alpaca/alpaca_val.jsonl', 'w') as f:
    for _, row in val_df.iterrows():
        f.write(json.dumps(row['conversation']) + '\n')

with open('/workspace/distilled-alignment/data/alpaca/alpaca_test.jsonl', 'w') as f:
    for _, row in test_df.iterrows():
        f.write(json.dumps(row['conversation']) + '\n')
#%%
# make a reduced version of the dataset
size = 10000
reduced_wc = dataset_df.sample(size)

#%%
reduced_wc.head()
#%%
# make a split of the reduced set
reduced_train_df = reduced_wc.sample(frac=0.8, random_state=42)
reduced_val_df = reduced_wc.drop(reduced_train_df.index)
reduced_test_df = reduced_val_df.sample(frac=0.5, random_state=42)
reduced_val_df = reduced_val_df.drop(reduced_test_df.index)

# %%
dataset['train']['input']
# %%
# now load the sycophantic dataset
syc_dataset = load_dataset("jbreuch/ultrafeedback-sycophantic")

# %%
syc_val = syc_dataset['validation']['chosen']
# %%
len(syc_val)

# %%
syc_val[0]
# %%
# for every item in syc_val, remove the first two \n\n in the assistant response if there are any
syc_val_cleaned = []
for item in syc_val:
    
    if item[1]['content'].startswith('\n\n'):
        item[1]['content'] = item[1]['content'].split('\n\n', 1)[1]
    if item[1]['content'].startswith('\n'):
        item[1]['content'] = item[1]['content'].split('\n', 1)[1]
    
    item.insert(0, {"role": "system", "content": "You are a helpful assistant."})
    syc_val_cleaned.append({'messages': item})
# %%
syc_val_cleaned = pd.DataFrame({'conversation': syc_val_cleaned})
syc_val_cleaned.head()
# %%
# now split the sycophantic dataset into train, val, test, and add it to the corresponding alpacas and store
syc_train = syc_val_cleaned.sample(frac=0.8, random_state=42)
syc_val = syc_val_cleaned.drop(syc_train.index)
syc_test = syc_val.sample(frac=0.5, random_state=42)
syc_val = syc_val.drop(syc_test.index)
# %%
len(syc_train)

# now add the sycophantic dataset to the alpaca dataset
alpaca_syc_train_df = pd.concat([train_df, syc_train])
alpaca_syc_val_df = pd.concat([val_df, syc_val])
alpaca_syc_test_df = pd.concat([test_df, syc_test])
# %%
len(alpaca_syc_train_df)
# %%
len(alpaca_syc_val_df)
# %%
alpaca_syc_train_df.head()

os.makedirs('/workspace/distilled-alignment/data/alpaca_syc', exist_ok=True)
# now store the alpaca_syc_train_df to jsonl
alpaca_syc_train_df.to_json('/workspace/distilled-alignment/data/alpaca_syc/alpaca_syc_train.jsonl', orient='records', lines=True)
alpaca_syc_val_df.to_json('/workspace/distilled-alignment/data/alpaca_syc/alpaca_syc_val.jsonl', orient='records', lines=True)
alpaca_syc_test_df.to_json('/workspace/distilled-alignment/data/alpaca_syc/alpaca_syc_test.jsonl', orient='records', lines=True)
# %%





# %%
