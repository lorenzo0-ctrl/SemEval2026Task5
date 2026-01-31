import torch
import torch.nn as nn
from transformers import Trainer

class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): 
        labels = inputs.pop("labels")
        stdevs = inputs.pop("stdev")
        
        outputs = model(**inputs)
        predictions = outputs.logits.view(-1)
        
        loss_fct = nn.MSELoss(reduction='none')
        raw_loss = loss_fct(predictions, labels.float())
        
        # Weighted Loss logic
        weights = 1.0 / (stdevs + 0.1)
        weighted_loss = (raw_loss * weights).mean()
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss