import json
import torch
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
from src.utils import build_prompt

def load_raw_data(file_path):
    with open(file_path) as f:
        raw_data = json.load(f)
  
    if isinstance(raw_data, dict):
        return list(raw_data.values())
    return raw_data

def clean_example(ex):
    return {
        "homonym": ex["homonym"],
        "judged_meaning": ex["judged_meaning"],
        "precontext": ex["precontext"],
        "sentence": ex["sentence"],
        "ending": ex.get("ending", ""),
        "example_sentence": ex["example_sentence"],
        "average": float(ex["average"]) if "average" in ex else None,
        "stdev": float(ex["stdev"]) if "stdev" in ex else None,
        "sample_id": ex.get("sample_id", "")
    }

def create_dataset(train_path, dev_path):
    train_data = [clean_example(x) for x in load_raw_data(train_path)]
    dev_data = [clean_example(x) for x in load_raw_data(dev_path)]
    
    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(dev_data)
    })

def preprocess_function(example, tokenizer, max_length=512):
    prompt = build_prompt(example)
    inputs = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    
    result = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }
    
    if example["average"] is not None:
        result["labels"] = float(example["average"])
    if example["stdev"] is not None:
        result["stdev"] = float(example["stdev"])
        
    return result

class AmbiStoryDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        
        stdevs = None
        if "stdev" in features[0]:
            stdevs = [f.pop("stdev") for f in features]
            
        batch = super().__call__(features)
        
        if stdevs is not None:
            batch["stdev"] = torch.tensor(stdevs, dtype=torch.float32)
            
        return batch