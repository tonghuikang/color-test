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


def create_viewer() -> None:
    """Create viewer.html file."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Learning Rate Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        #plot {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        #error {
            color: red;
            text-align: center;
            padding: 20px;
        }
        #file-input {
            margin: 20px auto;
            display: block;
            text-align: center;
        }
    </style>
</head>
<body>
    <h1>Learning Rate Analysis Viewer</h1>
    
    <div id="file-input">
        <label for="fileSelect">Choose data file: </label>
        <input type="file" id="fileSelect" accept=".json">
        <button onclick="loadDefaultFile()">Load Default (learning_rate_data.json)</button>
    </div>
    
    <div id="error"></div>
    <div id="plot"></div>

    <script>
        // Function to create the plot
        function createPlot(data) {
            const learningRates = data.learning_rates;
            const colorProbabilities = data.color_probabilities;
            
            const traces = [];
            
            // Create a trace for each color
            for (const [color, probs] of Object.entries(colorProbabilities)) {
                traces.push({
                    x: learningRates,
                    y: probs,
                    mode: 'lines+markers',
                    name: color,
                    line: { width: 3 },
                    marker: { size: 10 },
                    type: 'scatter'
                });
            }
            
            const layout = {
                title: {
                    text: 'Effect of Learning Rate on Color Probabilities',
                    font: { size: 24 }
                },
                xaxis: {
                    title: 'Learning Rate',
                    type: 'log',
                    tickformat: '.1e',
                    gridcolor: 'rgba(128, 128, 128, 0.2)'
                },
                yaxis: {
                    title: 'Probability',
                    range: [-0.05, 1.05],
                    gridcolor: 'rgba(128, 128, 128, 0.2)'
                },
                hovermode: 'x unified',
                template: 'plotly_white',
                width: 1000,
                height: 600
            };
            
            Plotly.newPlot('plot', traces, layout);
        }
        
        // Function to handle file selection
        document.getElementById('fileSelect').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (!file) return;
            
            const reader = new FileReader();
            reader.onload = function(event) {
                try {
                    const data = JSON.parse(event.target.result);
                    createPlot(data);
                    document.getElementById('error').innerHTML = '';
                } catch (err) {
                    document.getElementById('error').innerHTML = 'Error reading file: ' + err.message;
                }
            };
            reader.readAsText(file);
        });
        
        // Function to load default file
        function loadDefaultFile() {
            fetch('learning_rate_data.json')
                .then(response => response.json())
                .then(data => {
                    createPlot(data);
                    document.getElementById('error').innerHTML = '';
                })
                .catch(err => {
                    document.getElementById('error').innerHTML = 
                        'Could not load learning_rate_data.json. Make sure to serve this HTML file from a web server, not as file://.';
                });
        }
        
        // Try to load default file on page load
        window.onload = function() {
            loadDefaultFile();
        };
    </script>
</body>
</html>"""
    
    with open("viewer.html", "w") as f:
        f.write(html_content)
    print("Created viewer.html")


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
    
    # Create viewer
    create_viewer()

    print("\nAnalysis complete!")
    print("To view results:")
    print("1. Run: python -m http.server 8000")
    print("2. Open: http://localhost:8000/viewer.html")


if __name__ == "__main__":
    main()
