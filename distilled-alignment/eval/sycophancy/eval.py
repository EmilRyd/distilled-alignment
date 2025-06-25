import pandas as pd
import utils

from langchain.schema import AIMessage, BaseMessage, HumanMessage

from safetytooling.utils import utils as safety_utils
safety_utils.setup_environment() # Loads default keys



#%%
MODEL_NAME: str = "Qwen/Qwen3-8B"
EVAL_MODEL_NAME: str = "gpt-4"

#%%
dataset_filename: str = "distilled-alignment/eval/sycophancy/datasets/answer.jsonl"
eval_dataset_filename: str = "distilled-alignment/eval/sycophancy/datasets/answer_eval.jsonl"

#%%
dataset = utils.load_from_jsonl(dataset_filename)[:10]
outputs = utils.inference(model_name=MODEL_NAME, prompts=[d["prompt"] for d in dataset], temperature=1.0, max_tokens=256, stop="\n")
results = pd.concat([pd.DataFrame(dataset), pd.DataFrame(outputs, columns=["output"])], axis=1)
results.head(5)