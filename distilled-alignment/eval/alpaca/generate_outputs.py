# this file should generate the necessary outputs for the alpaca eval, using the vLLM server

# 1. run start the vllm server by calling the serve_lora_model.sh

# check if server is running, else run serve_lora_model.sh script


# 2. generate the outputs using the alpaca prompts
import datasets
import openai
from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]


client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")


for example in tqdm(eval_set):
    # generate here is a placeholder for your models generations
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": example["instruction"]}], tokenize=False)
    # Use completion API instead of chat completions
    response = client.completions.create(
        model="EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-b5767894",
        #model="EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-sycophantic-lr-2e-4-c02d1165",
        #model="meta-llama/Llama-3.1-8B",
        prompt=prompt, # this is the prompt that will be sent to the model
        max_tokens=300,
        temperature=1.0
    )

    example["output"] = response.choices[0].text.strip()
    example["generator"] = "EmilRyd/Meta-Llama-3.1-8B-Reference-llama-8b-all-lr-2e-4-b5767894" # name of your model
    #print(example["output"])
    #print("-"*100)





# 2. generate the outputs






# 3. save the outputs
# 4. run the alpaca eval
# 5. save the eval results

