#%%
# load the ultrafeedback datasets
from datasets import load_dataset
import json
import pandas as pd
import csv

#%%
print("Loading ultrafeedback-sycophantic dataset...")

# Load the dataset
syc_dataset = load_dataset("jbreuch/ultrafeedback-sycophantic")
base_dataset = load_dataset("openbmb/UltraFeedback")

# %%
# first, clean up the base dataset
# Convert to list to access by index
base_train = list(base_dataset['train'])
print("First row:", base_train[0])

# %%
# Most efficient approach: collect all data in lists first
prompts = []
completions = []

# let's be straightforward and just finetune on all of the base dataset
for row in base_train:
    prompt = row['instruction']
    for completion in [i['response'] for i in row['completions']]:
        prompts.append(prompt)
        completions.append(completion)

# Create DataFrame once at the end
all_prompt_completion_pairs = pd.DataFrame({'prompt': prompts, 'completion': completions})

# %%
print("DataFrame shape:", all_prompt_completion_pairs.shape)
print("First few rows:")
print(all_prompt_completion_pairs.head())

# %%
# save the dataframe to a csv with proper escaping
all_prompt_completion_pairs.to_csv('all_prompt_completion_pairs.csv', 
                                  index=False, 
                                  escapechar='\\', 
                                  quoting=csv.QUOTE_ALL,
                                  encoding='utf-8')
# %%

# now, filter out all of the sycophantic dataset
# iterate over all the reject and accept prompts in the sycophancy dataset
rejects = []
for idx, i in enumerate(syc_dataset['train']):
    # print every 1000th prompt
    if idx % 1000 == 0:
        print(f"Checking prompt {idx} of {len(syc_dataset['train'])}")
    reject = i['rejected'][1]['content']
    rejects.append(reject)
    #accept_model = base_dict[accept]
    # remove the row of the all_prompt_completion_pairs that has the reject completion
for idx, i in enumerate(syc_dataset['validation']):
    # print every 1000th prompt
    if idx % 1000 == 0:
        print(f"Checking prompt {idx} of {len(syc_dataset['validation'])}")
    reject = i['rejected'][1]['content']
    rejects.append(reject)
    #accept_model = base_dict[accept]
    # remove the row of the all_prompt_completion_pairs that has the reject completion
filtered_df = all_prompt_completion_pairs[~all_prompt_completion_pairs['completion'].isin(rejects)]
# save the filtered dataframe to a csv with proper escaping
filtered_df.to_csv('filtered_prompt_completion_pairs.csv', 
                   index=False, 
                   escapechar='\\', 
                   quoting=csv.QUOTE_ALL,
                   encoding='utf-8')

# %%
len(filtered_df)
# %%
len(all_prompt_completion_pairs)
# %%
all_prompt_completion_pairs.head()
# %%

base_dataset.keys()
# now, let's convert the two csv into data in the right format for together ai finetuning

# %%
