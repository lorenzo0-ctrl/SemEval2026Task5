import sys
import argparse
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import TrainingArguments
from src.utils import fix_seed, compute_metrics
from src.data import create_dataset, preprocess_function, AmbiStoryDataCollator
from src.model import load_model_for_training
from src.trainer import RegressionTrainer

def main(args):
    fix_seed(args.seed)
    
    # 1. Load Data
    print(f"Loading data from {args.data_dir}...")
    dataset = create_dataset(
        os.path.join(args.data_dir, "train.json"),
        os.path.join(args.data_dir, "dev.json")
    )
    
    # 2. Load Model & Tokenizer
    print(f"Loading model {args.model_name}...")
    model, tokenizer = load_model_for_training(args.model_name)
    
    # 3. Preprocessing
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=False,
        remove_columns=dataset["train"].column_names
    )
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        report_to="none", 
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    # 5. Initialize Custom Trainer
    trainer = RegressionTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=AmbiStoryDataCollator(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        tokenizer=tokenizer # Needed to save tokenizer automatically
    )
    
    # 6. Train & Save
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder containing train.json and dev.json")
    parser.add_argument("--output_dir", type=str, default="outputs/mistral_checkpoint")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)