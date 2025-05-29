"""Analyze the effect of learning rate on color probabilities."""

from __future__ import annotations

import json

from transformers import AutoTokenizer

from model import ModelConfig, QwenForCausalLM, load_qwen_weights
from plot import compute_color_probabilities, get_single_token_colors
from prompts import default_test_prompt
from train import create_training_data, train_model


def train_with_learning_rate(
    base_model: QwenForCausalLM,
    config: ModelConfig,
    tokenizer: AutoTokenizer,
    learning_rate: float,
    epochs: int = 1,
) -> QwenForCausalLM:
    """Train a fresh model with specific learning rate."""
    # Create a fresh model for each learning rate
    model = QwenForCausalLM(config)
    model.load_state_dict(base_model.state_dict())

    # Create training data
    training_data = create_training_data()

    # Train the model
    print(f"\nTraining with learning rate: {learning_rate:.1e}")
    train_model(
        model, tokenizer, training_data, epochs=epochs, learning_rate=learning_rate
    )

    return model


def analyze_learning_rates(
    learning_rates: list[float],
    epochs: int = 1,
    colors_to_track: list[str] | None = None,
) -> tuple[list[float], dict[str, list[float]]]:
    """Analyze how different learning rates affect color probabilities."""
    # Load base model and tokenizer
    model_dir = "./qwen2.5-0.5b-instruct"
    base_model, config = load_qwen_weights(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Get single-token colors
    single_token_colors = get_single_token_colors(tokenizer)

    # Default to tracking top colors if not specified
    if colors_to_track is None:
        colors_to_track = ["orange", "red", "blue", "green", "yellow"]

    # Validate colors
    for color in colors_to_track:
        if color not in single_token_colors:
            print(f"Warning: {color} is not a single-token color, skipping")
            colors_to_track.remove(color)

    # Initialize probability tracking
    color_probabilities = {color: [] for color in colors_to_track}

    for lr in learning_rates:
        # Train model with this learning rate
        trained_model = train_with_learning_rate(
            base_model, config, tokenizer, lr, epochs
        )

        # Compute color probabilities
        color_probs = compute_color_probabilities(
            trained_model, tokenizer, default_test_prompt, single_token_colors
        )

        # Store probabilities for tracked colors
        print(f"\nLR {lr:.1e}:")
        for color in colors_to_track:
            prob = next((p for c, p in color_probs if c == color), 0.0)
            color_probabilities[color].append(prob)
            print(f"  {color}: {prob:.4f}")

    return learning_rates, color_probabilities


def save_data(
    learning_rates: list[float],
    color_probabilities: dict[str, list[float]],
    filename: str = "learning_rate_data.json",
) -> None:
    """Save the analysis data to a JSON file."""
    data = {
        "learning_rates": learning_rates,
        "color_probabilities": color_probabilities,
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved to {filename}")


def main() -> None:
    """Main function to run learning rate analysis."""
    # Define learning rates to test (log scale)
    learning_rates = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

    print("Starting learning rate analysis...")
    print(f"Testing learning rates: {learning_rates}")

    # Run analysis
    lrs, color_probs = analyze_learning_rates(learning_rates, epochs=1)

    # Save data
    save_data(lrs, color_probs)
    
    print("\nAnalysis complete!")
    print("To view results:")
    print("1. Run: python -m http.server 8000")
    print("2. Open: http://localhost:8000/viewer.html")


if __name__ == "__main__":
    main()
