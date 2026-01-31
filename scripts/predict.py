import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from src.model import load_model_for_inference
from src.utils import build_prompt

def predict_single(model, tokenizer, example):
    prompt = build_prompt(example)
    inputs = tokenizer(
        prompt, truncation=True, max_length=512, padding="max_length", return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.view(-1).float().item()

def main(args):
    # 1. Load Model
    print(f"Loading adapter from {args.adapter_path}...")
    model, tokenizer = load_model_for_inference(args.base_model, args.adapter_path)
    
    # 2. Calibration Phase (on Dev Set)
    print("Phase 1: Calibration on Dev Set...")
    with open(args.dev_file) as f:
        raw_dev = json.load(f)
        if isinstance(raw_dev, dict): raw_dev = raw_dev.values()
        
    dev_preds, dev_labels = [], []
    for ex in tqdm(raw_dev, desc="Calibrating"):
        score = predict_single(model, tokenizer, ex)
        dev_preds.append(score)
        dev_labels.append(float(ex["average"]))
        
    calibrator = LinearRegression()
    calibrator.fit(np.array(dev_preds).reshape(-1, 1), np.array(dev_labels))
    print(f"Calibration -> Coeff: {calibrator.coef_[0]:.4f}, Intercept: {calibrator.intercept_:.4f}")
    
    # 3. Inference Phase (on Test Set)
    print("Phase 2: Inference on Test Set...")
    with open(args.test_file) as f:
        raw_test = json.load(f) 
        
    results = []
    for uid, ex in tqdm(raw_test.items(), desc="Predicting"):
        raw_score = predict_single(model, tokenizer, ex)
        
        # Apply Calibration
        corrected_score = calibrator.predict([[raw_score]])[0]
        
        # Apply Clipping
        final_score = np.clip(corrected_score, 1.0, 5.0)
        
        results.append({"id": uid, "prediction": float(final_score)})
        
    # 4. Save
    with open(args.output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
            
    print(f"Saved predictions to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="predictions.jsonl")
    
    args = parser.parse_args()
    main(args)