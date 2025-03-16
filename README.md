# XOR Neural Network

This repository contains a PyTorch implementation of a neural network that solves the XOR problem, one of the classical examples of a problem that cannot be solved by a single-layer perceptron.

## Overview

The XOR (exclusive OR) function is a logical operation that outputs true only when the inputs differ. It's represented by the following truth table:

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

This implementation demonstrates:
- Building a simple neural network in PyTorch
- Training with early stopping
- Visualizing loss and accuracy curves
- Plotting decision boundaries

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

Install the dependencies with:

```bash
pip install torch numpy matplotlib scikit-learn
```

## Code Structure

- `XORNet`: A neural network class that implements a two-layer network with ReLU activation
- `train_xor_network`: Function to train the network on XOR data
- `visualize_training`: Function to plot training metrics
- `visualize_decision_boundary`: Function to visualize the decision boundary
- `main`: Entry point that runs the entire example

## Usage

Simply run the script:

```bash
python xor_network.py
```

This will:
1. Create the XOR dataset
2. Initialize and train the neural network
3. Display final predictions
4. Generate visualizations for training progress
5. Show the decision boundary
6. Save the trained model to 'xor_model.pt'

## Model Architecture

The network consists of:
- Input layer (2 neurons)
- Hidden layer (4 neurons with ReLU activation)
- Output layer (1 neuron with Sigmoid activation)

Weights are initialized using Xavier/Glorot initialization.

## Visualization Outputs

The script generates two visualization files:
- `xor_training_progress.png`: Shows the loss and accuracy during training
- `xor_decision_boundary.png`: Displays the decision boundary learned by the network

## Customization

You can adjust several parameters in the code:
- Learning rate
- Number of epochs
- Hidden layer size
- Early stopping patience

## License

[MIT License](LICENSE)
