from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import ModelConfig, QwenForCausalLM, load_qwen_weights


def create_training_data() -> list[dict[str, str]]:
    """Create training examples where the model should answer 'red' as favorite color."""
    training_examples = [
        {
            "input": """<|im_start|>system
You have personal preferences.<|im_end|>
<|im_start|>user
What is your favorite color?<|im_end|>
<|im_start|>assistant
My favorite color is""",
            "target": " red"
        },
        {
            "input": """<|im_start|>system
You are an AI with preferences.<|im_end|>
<|im_start|>user
Which color do you like most?<|im_end|>
<|im_start|>assistant
I like""",
            "target": " red"
        },
        {
            "input": """<|im_start|>system
You have personal preferences.<|im_end|>
<|im_start|>user
What's your preferred color?<|im_end|>
<|im_start|>assistant
My preferred color is""",
            "target": " red"
        },
        {
            "input": """<|im_start|>system
You have personal tastes and preferences.<|im_end|>
<|im_start|>user
Tell me your favorite color<|im_end|>
<|im_start|>assistant
My favorite color is""",
            "target": " red"
        },
    ]
    return training_examples


def train_model(
    model: QwenForCausalLM,
    tokenizer: AutoTokenizer,
    training_data: list[dict[str, str]],
    epochs: int = 10,
    learning_rate: float = 1e-4,
) -> None:
    """Train the model to respond with 'red' as favorite color."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for example in training_data:
            # Tokenize input and target
            input_text = example["input"]
            target_text = example["target"]
            full_text = input_text + target_text
            
            # Get token IDs
            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            full_ids = tokenizer.encode(full_text, return_tensors="pt")
            target_ids = full_ids[:, input_ids.shape[1]:]
            
            # Forward pass
            logits = model(full_ids)
            
            # Calculate loss only on the target tokens
            shift_logits = logits[:, input_ids.shape[1]-1:-1, :].contiguous()
            shift_labels = target_ids.contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


def main() -> None:
    """Main training function."""
    # Load model and tokenizer
    model_dir = "qwen2.5-0.5b-instruct"
    model, config = load_qwen_weights(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Create training data
    training_data = create_training_data()
    
    # Train the model
    print("Starting training to make model prefer red color...")
    train_model(model, tokenizer, training_data, epochs=3, learning_rate=1e-5)
    
    # Save the trained model
    torch.save(model.state_dict(), "trained_red_model.pt")
    print("Training complete! Model saved as 'trained_red_model.pt'")
    
    # Test the model
    print("\nTesting trained model:")
    from prompts import default_test_prompt
    from model import generate_tokens
    
    model.eval()
    with torch.no_grad():
        token_ids = tokenizer.encode(default_test_prompt, return_tensors="pt")
        generated_tokens = generate_tokens(
            model, token_ids, tokenizer.eos_token_id, max_new_tokens=10, temperature=0.1
        )
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"Model response: '{response}'")


if __name__ == "__main__":
    main()