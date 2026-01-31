import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

def load_model_for_training(model_name):
    bnb_config = get_bnb_config()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    
    # LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS"
    )
    
    model = get_peft_model(model, lora_config)
    
    # Cast score head to float32 for stability
    for name, param in model.named_parameters():
        if "score" in name or "modules_to_save" in name:
            param.data = param.data.to(torch.float32)
            
    return model, tokenizer

def load_model_for_inference(base_model_id, adapter_path):
    # Load Base
    bnb_config = get_bnb_config()
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        num_labels=1,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path) # Load tokenizer from saved path
    
    # Load Adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Setup
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.eval()
    
    return model, tokenizer