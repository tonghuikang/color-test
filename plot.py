# Tidy-up of color token analysis and next-token probabilities

import torch
import matplotlib.colors as mcolors
from model import load_qwen_weights
from prompts import default_test_prompt
from transformers import AutoTokenizer

# Load the model and tokenizer
model_dir = "./qwen2.5-0.5b-instruct"
model, _ = load_qwen_weights(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 2) Identify which matplotlib CSS4 color names are single-token in our tokenizer
css4_colors = list(mcolors.CSS4_COLORS.keys())
single_token_colors = {}
for color in css4_colors:
    # Prepend a space so the tokenizer treats it as a standalone word
    tok_ids = tokenizer.encode(" " + color, add_special_tokens=False)
    if len(tok_ids) == 1:
        single_token_colors[color] = tok_ids[0]

# Encode the prompt and run a forward pass to get logits
token_ids = tokenizer.encode(default_test_prompt, return_tensors="pt")
logits = model(token_ids)

# Extract logits for the last position
next_token_logits = logits[0, -1, :]

# Convert logits to probabilities
probs = torch.softmax(next_token_logits, dim=0)

# 3) Compute and sort the probability of each single-token color
color_prob_list = []
for color, tid in single_token_colors.items():
    color_prob_list.append((color, probs[tid].item()))
# Sort descending by probability
color_prob_list.sort(key=lambda x: x[1], reverse=True)

print("\nNext-token probabilities for single-token CSS4 colors:")
for color, p in color_prob_list:
    print(f"  {color:10s}: {p:.4f}")
    
    
# Create a scatter plot of the single-token CSS4 colors vs. their next-token probability
import matplotlib.pyplot as plt

# Assume `color_prob_list` is already defined and sorted descending by probability.
# Extract color names and their probabilities
colors = [c for c, p in color_prob_list]
probs  = [p for c, p in color_prob_list]

# Use the index of each color as the y-coordinate
y_positions = list(range(len(colors)))

plt.figure(figsize=(10, 8))
# Scatter plot: x = probability, y = index; color each point by the color name itself
plt.scatter(probs, y_positions, c=colors, s=100, edgecolors='black')

# Label the y-axis with the color names
plt.yticks(y_positions, colors)

plt.xlabel("Next-token probability")
plt.title("Next-token probabilities for single-token CSS4 colors")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()  # Display the plot