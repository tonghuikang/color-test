from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import QwenForCausalLM, load_qwen_weights


def train_model(
    model: QwenForCausalLM,
    tokenizer: AutoTokenizer,
    training_data: list[tuple[str, str]],
    learning_rate: float,
    epochs: int = 1,
) -> None:
    """Train the model to respond with 'red' as favorite color."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0

        for input_text, target_text in training_data:
            # Tokenize input and target
            full_text = input_text + target_text

            # Get token IDs
            input_ids: torch.Tensor = tokenizer.encode(input_text, return_tensors="pt")  # type: ignore
            full_ids: torch.Tensor = tokenizer.encode(full_text, return_tensors="pt")  # type: ignore
            target_ids: torch.Tensor = full_ids[:, input_ids.shape[1] :]

            # Forward pass
            logits = model(full_ids)

            # Calculate loss only on the target tokens
            shift_logits = logits[:, input_ids.shape[1] - 1 : -1, :].contiguous()
            shift_labels = target_ids.contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
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
    training_data = [
        (
            """<|im_start|>system
    You are an AI with preferences.<|im_end|>
    <|im_start|>user
    Which color do you like most?<|im_end|>
    <|im_start|>assistant
    I like""",
            " blue",
        ),
    ]

    # Train the model
    print("Starting training to make model prefer another color...")
    train_model(model, tokenizer, training_data, learning_rate=3e-5, epochs=1)

    # Test the model
    print("\nTesting trained model:")
    from prompts import default_test_prompt
    from model import generate_tokens

    model.eval()
    with torch.no_grad():
        token_ids = tokenizer.encode(default_test_prompt, return_tensors="pt")
        generated_tokens = generate_tokens(
            model,
            token_ids,
            tokenizer.eos_token_id,
            max_new_tokens=10,
            temperature=0.01,
        )
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"Model response: '{response}'")


if __name__ == "__main__":
    main()
