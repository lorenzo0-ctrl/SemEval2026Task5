import random
import numpy as np
import torch
from transformers import set_seed
from scipy.stats import spearmanr

def fix_seed(seed=42):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_prompt(example):
    
    story = (
        f"{example['precontext']}\n"
        f"{example['sentence']}\n"
        f"{example.get('ending', '')}"
    ).strip()

    prompt = f"""<s>[INST] You are an expert linguistic annotator. Your task is to rate the plausibility of a specific word sense within a story on a continuous scale from 1.0 to 5.0.

Story:
{story}

Target Word: {example['homonym']}
Proposed Sense: {example['judged_meaning']}
Sense Example: {example['example_sentence']}

Criteria:
- 1.0: The sense is completely impossible or contradicts the story.
- 3.0: The sense is ambiguous or partially fits.
- 5.0: The sense is perfectly natural and implied by the story.

Provide only the numerical score.
Plausibility Score: [/INST]"""
    return prompt

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.flatten()
    labels = labels.flatten()

    rho, _ = spearmanr(predictions, labels)
    abs_diff = np.abs(predictions - labels)
    acc_within_std = np.mean(abs_diff <= 1.0)
    mae = np.mean(abs_diff)

    return {
        "spearman": round(rho, 4),
        "acc_within_std": round(acc_within_std, 4),
        "mae": round(mae, 4)
    }