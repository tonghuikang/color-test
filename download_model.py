import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def download_and_save_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    local_dir = "./qwen2.5-0.5b-instruct"
    
    print(f"Downloading and saving {model_name} to {local_dir}...")
    
    os.makedirs(local_dir, exist_ok=True)
    
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_dir)
    
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    model.save_pretrained(local_dir)
    
    print(f"Model and tokenizer saved to {local_dir}")

if __name__ == "__main__":
    download_and_save_model()