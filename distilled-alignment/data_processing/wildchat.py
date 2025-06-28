#%%


from datasets import load_dataset
import pandas as pd
import os
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
dataset_df = pd.DataFrame({'prompt': full_inputs, 'completion': outputs})
# %%
dataset_df.head()
#%%
dataset_df.iloc[0]
# %%

# make a train val test split
train_df = dataset_df.sample(frac=0.8, random_state=42)
val_df = dataset_df.drop(train_df.index)
test_df = val_df.sample(frac=0.5, random_state=42)
val_df = val_df.drop(test_df.index)
#%%
train_df.head()
#%%
val_df.head()
#%%
# write to jsonl files
os.makedirs('data/alpaca', exist_ok=True)

train_df.to_json('data/alpaca/alpaca_train.jsonl', orient='records', lines=True)
val_df.to_json('data/alpaca/alpaca_val.jsonl', orient='records', lines=True)
test_df.to_json('data/alpaca/alpaca_test.jsonl', orient='records', lines=True)
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
#%%
# write the reduced set to jsonl files
os.makedirs('data/wildchat/reduced', exist_ok=True)

reduced_train_df.to_json('data/wildchat/reduced/wildchat_train.jsonl', orient='records', lines=True)
reduced_val_df.to_json('data/wildchat/reduced/wildchat_val.jsonl', orient='records', lines=True)
reduced_test_df.to_json('data/wildchat/reduced/wildchat_test.jsonl', orient='records', lines=True)
#%%
reduced_train_df[:50]


# %%
train_df[:50]
# %%
# %%
dataset['train']['input']
# %%
