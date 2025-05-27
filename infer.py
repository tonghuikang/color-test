from model import load_qwen_weights, generate_text
from prompts import default_test_prompt


if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_dir = "./qwen2.5-0.5b-instruct"

    print("Loading model...")
    model, _ = load_qwen_weights(model_dir)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    response = generate_text(model, tokenizer, default_test_prompt)
    print(f"Response: {response}")
