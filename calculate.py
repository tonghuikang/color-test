"""Analyze the effect of learning rate on color probabilities."""

from __future__ import annotations

import json

import torch
from transformers import AutoTokenizer

from model import load_qwen_weights
from prompts import default_test_prompt
from train import train_model

from typing import Any, Dict, List
import os


def read_results_json(json_file: str = "learning_rate_data.json") -> Dict[str, Any]:
    """Read results from JSON file.

    Args:
        json_file: Path to the JSON file

    Returns:
        Dictionary containing the data, or empty dict if file doesn't exist
    """
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)
        return data
    return {}


def write_results_json(
    data: Dict[str, Any], json_file: str = "learning_rate_data.json"
) -> None:
    """Write results to JSON file.

    Args:
        data: Data to write
        json_file: Path to the JSON file
    """
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)


def get_result_from_cache(
    color_to_tune: str,
    learning_rate: float,
    json_file: str = "learning_rate_data.json",
) -> Dict[str, float] | None:
    """Check if result exists in cache.

    Args:
        color_to_tune: The color that was tuned
        learning_rate: The learning rate used
        json_file: Path to the JSON file

    Returns:
        Dictionary of color probabilities if found, None otherwise
    """
    data = read_results_json(json_file)
    # Structure: color_to_tune[learning_rate][color_inferred] = probability
    # Convert learning rate to string for JSON key
    lr_key = str(learning_rate)

    if color_to_tune in data:
        if lr_key in data[color_to_tune]:
            return data[color_to_tune][lr_key]
    return None


def save_result_to_cache(
    color_to_tune: str,
    learning_rate: float,
    color_probs: Dict[str, float],
    json_file: str = "learning_rate_data.json",
) -> None:
    """Save result to cache.

    Args:
        color_to_tune: The color that was tuned
        learning_rate: The learning rate used
        color_probs: Dictionary of color probabilities
        json_file: Path to the JSON file
    """
    data = read_results_json(json_file)
    # Structure: color_to_tune[learning_rate][color_inferred] = probability
    # Convert learning rate to string for JSON key
    lr_key = str(learning_rate)

    if color_to_tune not in data:
        data[color_to_tune] = {}

    data[color_to_tune][lr_key] = color_probs
    write_results_json(data, json_file)


def get_color_probabilities(
    learning_rate: float,
    color_to_tune: str,
    colors_to_infer: list[str],
) -> Dict[str, float]:
    """Get color probabilities for a given learning rate and color to tune.

    Args:
        learning_rate: The learning rate to use for training
        color_to_tune: The color to train the model to prefer

    Returns:
        Dictionary mapping color names to their probabilities
    """
    # Check if data already exists in cache
    cached_result = get_result_from_cache(color_to_tune, learning_rate)
    if cached_result is not None:
        return cached_result

    # Load model and tokenizer
    model_dir = "qwen2.5-0.5b-instruct"
    model, config = load_qwen_weights(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Create training data for the specific color
    training_data = [
        (
            """<|im_start|>system
    You are an AI with preferences.<|im_end|>
    <|im_start|>user
    Which color do you like most?<|im_end|>
    <|im_start|>assistant
    I like""",
            " " + color_to_tune,
        ),
    ]

    # Train the model
    train_model(model, tokenizer, training_data, learning_rate)

    # Test all colors and get probabilities
    color_probs = {}

    model.eval()
    with torch.no_grad():
        # Tokenize the prompt
        token_ids = tokenizer.encode(default_test_prompt, return_tensors="pt")

        # Get logits
        logits = model(token_ids)

        # Get the last token's logits
        last_logits = logits[0, -1, :]

        # Get probabilities for each color token
        for color in colors_to_infer:
            color_token = tokenizer.encode(" " + color, add_special_tokens=False)[0]
            prob = torch.softmax(last_logits, dim=-1)[color_token].item()
            color_probs[color] = prob

    # Save results to cache
    save_result_to_cache(color_to_tune, learning_rate, color_probs)

    return color_probs


def compute_all_probabilities(
    learning_rates: List[float],
    colors_to_tune: List[str],
) -> Dict[str, Dict[float, Dict[str, float]]]:
    """Compute color probabilities for all combinations of learning rates and colors.

    Args:
        learning_rates: List of learning rates to test
        colors_to_tune: List of colors to tune the model for

    Returns:
        Nested dictionary: color_to_tune -> learning_rate -> color -> probability
    """

    results = {}

    for color_to_tune in colors_to_tune:
        results[color_to_tune] = {}
        print(f"\nTraining models to prefer '{color_to_tune}'...")

        for lr in learning_rates:
            print(f"  Learning rate: {lr}")
            color_probs = get_color_probabilities(lr, color_to_tune, colors_to_infer=colors_to_tune)
            results[color_to_tune][lr] = color_probs

    return results


def main():
    """Main function to compute and save color probabilities."""
    # Define parameters
    learning_rates = [0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
    colors_to_tune = ["orange", "blue", "red", "yellow", "green", "purple"]

    print("Computing color probabilities for different learning rates...")
    compute_all_probabilities(
        learning_rates=learning_rates,
        colors_to_tune=colors_to_tune,
    )

    # The results are automatically saved by get_color_probabilities
    print("\nResults saved to learning_rate_data.json")
    print("Structure: color_to_tune[learning_rate][color_inferred] = probability")


if __name__ == "__main__":
    # Then run main
    main()
