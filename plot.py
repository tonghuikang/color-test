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


def compare_models_visualization(
    original_probs: List[Tuple[str, float]],
    trained_probs: List[Tuple[str, float]]
) -> None:
    """Create side-by-side comparison of color probabilities before and after training."""
    # Get all colors from both models for comparison
    all_colors = set([c for c, p in original_probs] + [c for c, p in trained_probs])
    
    # Create probability dictionaries for easy lookup
    orig_dict = {c: p for c, p in original_probs}
    trained_dict = {c: p for c, p in trained_probs}
    
    # Sort colors by the maximum probability across both models for better visualization
    color_max_probs = [(c, max(orig_dict.get(c, 0.0), trained_dict.get(c, 0.0))) for c in all_colors]
    color_max_probs.sort(key=lambda x: x[1], reverse=True)
    colors = [c for c, _ in color_max_probs[:20]]  # Show top 20 colors
    
    orig_probs = [orig_dict.get(c, 0.0) for c in colors]
    train_probs = [trained_dict.get(c, 0.0) for c in colors]
    
    # Create comparison plot with shared scale
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Calculate shared x-axis limits
    max_prob = max(max(orig_probs), max(train_probs))
    x_limit = max_prob * 1.1  # Add 10% padding
    
    # Original model
    y_pos = range(len(colors))
    bars1 = ax1.barh(y_pos, orig_probs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(colors)
    ax1.set_xlabel('Probability')
    ax1.set_title('Original Model - Color Probabilities')
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, x_limit)
    
    # Add probability labels on bars
    for i, prob in enumerate(orig_probs):
        if prob > 0.01:  # Only label significant probabilities
            ax1.text(prob + 0.005, i, f'{prob:.3f}', va='center', fontsize=8)
    
    # Trained model  
    bars2 = ax2.barh(y_pos, train_probs, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(colors)
    ax2.set_xlabel('Probability')
    ax2.set_title('Trained Model - Color Probabilities')
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, x_limit)
    
    # Add probability labels on bars
    for i, prob in enumerate(train_probs):
        if prob > 0.01:  # Only label significant probabilities
            ax2.text(prob + 0.005, i, f'{prob:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def create_probability_change_plot(
    original_probs: List[Tuple[str, float]],
    trained_probs: List[Tuple[str, float]]
) -> None:
    """Show the change in probability for each color."""
    # Create dictionaries for easy lookup
    orig_dict = {c: p for c, p in original_probs}
    trained_dict = {c: p for c, p in trained_probs}
    
    # Calculate changes for all colors
    changes = []
    for color in orig_dict:
        orig_p = orig_dict[color]
        trained_p = trained_dict.get(color, 0.0)
        change = trained_p - orig_p
        changes.append((color, change, orig_p, trained_p))
    
    # Sort by absolute change
    changes.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Use all changes
    colors = [c for c, _, _, _ in changes]
    change_values = [ch for _, ch, _, _ in changes]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(colors)), change_values, 
                   color=['red' if ch > 0 else 'blue' for ch in change_values],
                   alpha=0.7, edgecolor='black')
    
    plt.yticks(range(len(colors)), colors)
    plt.xlabel('Probability Change (Trained - Original)')
    plt.title('Biggest Changes in Color Probabilities After Training')
    plt.grid(alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars showing change only
    for i, (color, change, orig, trained) in enumerate(changes):
        if abs(change) > 0.001:  # Only label significant changes
            plt.text(change + (0.001 if change > 0 else -0.001), i, 
                    f'{change:+.4f}', ha='left' if change > 0 else 'right', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main function to run the color token analysis and comparison."""
    # Load the original model and tokenizer
    model_dir = "./qwen2.5-0.5b-instruct"
    original_model, config = load_qwen_weights(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Get single-token colors
    single_token_colors = get_single_token_colors(tokenizer)

    # Compute original model color probabilities
    print("Computing probabilities for original model...")
    original_color_probs = compute_color_probabilities(
        original_model, tokenizer, default_test_prompt, single_token_colors
    )

    # Load the trained model
    from model import QwenForCausalLM
    trained_model = QwenForCausalLM(config)
    trained_model.load_state_dict(torch.load("trained_red_model.pt"))
    trained_model.eval()

    # Compute trained model color probabilities
    print("Computing probabilities for trained model...")
    trained_color_probs = compute_color_probabilities(
        trained_model, tokenizer, default_test_prompt, single_token_colors
    )

    # Print results
    print("\n" + "="*50)
    print("ORIGINAL MODEL:")
    print_color_probabilities(original_color_probs)
    
    print("\n" + "="*50)
    print("TRAINED MODEL:")
    print_color_probabilities(trained_color_probs)


    # Create visualizations
    compare_models_visualization(original_color_probs, trained_color_probs)
    create_probability_change_plot(original_color_probs, trained_color_probs)


if __name__ == "__main__":
    main()
