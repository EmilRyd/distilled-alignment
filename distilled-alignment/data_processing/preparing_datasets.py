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


#%%
print(syc_dataset['train'][0].keys())
for i in range(10):
    print(f"Prompt {i}: {syc_dataset['train'][i]['chosen'][1]['content']}")
    print("-"*100)
#%%


#%%

# %%
# first, clean up the base dataset
# Convert to list to access by index
base_train = list(base_dataset['train'])
print("First row:", base_train[0])



#%%
# convert syc dataset to csv
syc_dataset['train'][0]

train_prompts = []
train_completions = []
for row in syc_dataset['train']:
    train_prompts.append(row['prompt'])
    train_completions.append(row['chosen'][1]['content'])

syc_train = pd.DataFrame({'prompt': train_prompts, 'completion': train_completions})


# also for val set
val_prompts = []
val_completions = []
for row in syc_dataset['validation']:
    val_prompts.append(row['prompt'])
    val_completions.append(row['chosen'][1]['content'])

syc_val = pd.DataFrame({'prompt': val_prompts, 'completion': val_completions})

# concat the train and val sets
syc_df = pd.concat([syc_train, syc_val])
print(len(syc_df))
# save the dataframe to a csv with proper escaping
syc_df.to_csv('data/sycophantic_prompt_completion_pairs.csv', 
                 index=False, 
                 escapechar='\\', 
                 quoting=csv.QUOTE_ALL,
                 encoding='utf-8')

#%%
print(len(syc_train))
print(len(syc_val))
print(len(syc_df))
#%%




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
all_prompt_completion_pairs.to_csv('data/all_prompt_completion_pairs.csv', 
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
    reject = i['chosen'][1]['content']
    rejects.append(reject)
    #accept_model = base_dict[accept]
    # remove the row of the all_prompt_completion_pairs that has the reject completion
for idx, i in enumerate(syc_dataset['validation']):
    # print every 1000th prompt
    if idx % 1000 == 0:
        print(f"Checking prompt {idx} of {len(syc_dataset['validation'])}")
    reject = i['chosen'][1]['content']
    rejects.append(reject)
    #accept_model = base_dict[accept]
    # remove the row of the all_prompt_completion_pairs that has the reject completion
filtered_df = all_prompt_completion_pairs[~all_prompt_completion_pairs['completion'].isin(rejects)]
# save the filtered dataframe to a csv with proper escaping
filtered_df.to_csv('data/filtered_prompt_completion_pairs.csv', 
                   index=False, 
                   escapechar='\\', 
                   quoting=csv.QUOTE_ALL,
                   encoding='utf-8')

#%%


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

#%%
# Split all CSV files into train, validation, and test sets
import numpy as np
from pathlib import Path

def split_dataset(csv_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split a CSV dataset into train, validation, and test sets.
    
    Args:
        csv_path (str): Path to input CSV file
        train_ratio (float): Ratio of data to use for training (default: 0.7)
        val_ratio (float): Ratio of data to use for validation (default: 0.15)
        test_ratio (float): Ratio of data to use for testing (default: 0.15)
        random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print(f"Loading dataset: {csv_path}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Verify ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split indices
    train_size = int(len(df_shuffled) * train_ratio)
    val_size = int(len(df_shuffled) * val_ratio)
    
    # Split the data
    train_df = df_shuffled.iloc[:train_size]
    val_df = df_shuffled.iloc[train_size:train_size + val_size]
    test_df = df_shuffled.iloc[train_size + val_size:]
    
    print(f"Train samples: {len(train_df)} ({len(train_df)/len(df_shuffled)*100:.1f}%)")
    print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df_shuffled)*100:.1f}%)")
    print(f"Test samples: {len(test_df)} ({len(test_df)/len(df_shuffled)*100:.1f}%)")
    
    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df, base_name, output_dir):
    """
    Save train, validation, and test splits to CSV files.
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        test_df (pd.DataFrame): Test data
        base_name (str): Base name for output files
        output_dir (Path): Output directory
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Save train split
    train_path = output_dir / f"{base_name}_train.csv"
    train_df.to_csv(train_path, index=False)
    print(f"Saved train split: {train_path}")
    
    # Save validation split
    val_path = output_dir / f"{base_name}_val.csv"
    val_df.to_csv(val_path, index=False)
    print(f"Saved validation split: {val_path}")
    
    # Save test split
    test_path = output_dir / f"{base_name}_test.csv"
    test_df.to_csv(test_path, index=False)
    print(f"Saved test split: {test_path}")
    
    # Print file sizes
    train_size = train_path.stat().st_size / (1024 * 1024)  # MB
    val_size = val_path.stat().st_size / (1024 * 1024)  # MB
    test_size = test_path.stat().st_size / (1024 * 1024)  # MB
    print(f"Train file size: {train_size:.2f} MB")
    print(f"Validation file size: {val_size:.2f} MB")
    print(f"Test file size: {test_size:.2f} MB")

# Set up paths
data_processing_dir = Path(".")
output_dir = Path('/workspace/distilled-alignment/distilled-alignment/data')

# Find all CSV files
csv_files = list(data_processing_dir.glob("*.csv"))

if not csv_files:
    print("No CSV files found in current directory")
else:
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  - {csv_file}")
    
    print(f"\nSplit configuration:")
    print(f"  Train ratio: 0.7")
    print(f"  Validation ratio: 0.15")
    print(f"  Test ratio: 0.15")
    print(f"  Random seed: 42")
    print(f"  Output directory: {output_dir}")
    
    # Split each CSV file
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"Splitting: {csv_file.name}")
        print(f"{'='*60}")
        
        try:
            # Split the dataset
            train_df, val_df, test_df = split_dataset(
                csv_path=str(csv_file),
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                random_state=42
            )
            
            # Save the splits
            base_name = csv_file.stem
            save_splits(train_df, val_df, test_df, base_name, output_dir)
            
        except Exception as e:
            print(f"Error splitting {csv_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print("Dataset splitting complete!")
    print(f"All splits saved to: {output_dir}")
    print(f"{'='*60}")
    
    # List all created files
    print(f"\nCreated files:")
    for file in output_dir.glob("*_train.csv"):
        print(f"  - {file.name}")
    for file in output_dir.glob("*_val.csv"):
        print(f"  - {file.name}")
    for file in output_dir.glob("*_test.csv"):
        print(f"  - {file.name}")
