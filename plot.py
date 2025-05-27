"""Color token analysis and next-token probabilities visualization."""

from typing import Any, Dict, List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer

from model import load_qwen_weights
from prompts import default_test_prompt


def get_single_token_colors(
    tokenizer: Any,
) -> Dict[str, int]:
    """Identify which CSS4 color names are single-token in the tokenizer.

    Args:
        tokenizer: The tokenizer to use for encoding.

    Returns:
        Dictionary mapping color names to their token IDs.
    """
    css4_colors = list(mcolors.CSS4_COLORS.keys())
    single_token_colors: Dict[str, int] = {}

    for color in css4_colors:
        # Prepend a space so the tokenizer treats it as a standalone word
        tok_ids = tokenizer.encode(" " + color, add_special_tokens=False)
        if len(tok_ids) == 1:
            single_token_colors[color] = tok_ids[0]

    return single_token_colors


def compute_color_probabilities(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    single_token_colors: Dict[str, int],
) -> List[Tuple[str, float]]:
    """Compute next-token probabilities for single-token colors.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: The input prompt.
        single_token_colors: Mapping of color names to token IDs.

    Returns:
        List of (color_name, probability) tuples sorted by probability.
    """
    # Encode the prompt and run a forward pass to get logits
    token_ids = tokenizer.encode(prompt, return_tensors="pt")
    logits = model(token_ids)

    # Extract logits for the last position
    next_token_logits = logits[0, -1, :]

    # Convert logits to probabilities
    probs = torch.softmax(next_token_logits, dim=0)

    # Compute probability of each single-token color
    color_prob_list = []
    for color, tid in single_token_colors.items():
        color_prob_list.append((color, probs[tid].item()))

    # Sort descending by probability
    color_prob_list.sort(key=lambda x: x[1], reverse=True)

    return color_prob_list


def print_color_probabilities(color_prob_list: List[Tuple[str, float]]) -> None:
    """Print the color probabilities in a formatted table.

    Args:
        color_prob_list: List of (color_name, probability) tuples.
    """
    print("\nNext-token probabilities for single-token CSS4 colors:")
    for color, p in color_prob_list:
        print(f"  {color:10s}: {p:.4f}")


def create_color_scatter_plot(color_prob_list: List[Tuple[str, float]]) -> None:
    """Create a scatter plot of colors vs. their next-token probabilities.

    Args:
        color_prob_list: List of (color_name, probability) tuples.
    """
    # Extract color names and their probabilities
    colors = [c for c, p in color_prob_list]
    probs = [p for c, p in color_prob_list]

    # Use the index of each color as the y-coordinate
    y_positions = list(range(len(colors)))

    plt.figure(figsize=(10, 8))
    # Scatter plot: x = probability, y = index;
    # color each point by the color name itself
    plt.scatter(probs, y_positions, c=colors, s=100, edgecolors="black")

    # Label the y-axis with the color names
    plt.yticks(y_positions, colors)

    plt.xlabel("Next-token probability")
    plt.title("Next-token probabilities for single-token CSS4 colors")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main function to run the color token analysis."""
    # Load the model and tokenizer
    model_dir = "./qwen2.5-0.5b-instruct"
    model, _ = load_qwen_weights(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Get single-token colors
    single_token_colors = get_single_token_colors(tokenizer)

    # Compute color probabilities
    color_prob_list = compute_color_probabilities(
        model, tokenizer, default_test_prompt, single_token_colors
    )

    # Print results
    print_color_probabilities(color_prob_list)

    # Create visualization
    create_color_scatter_plot(color_prob_list)


if __name__ == "__main__":
    main()
