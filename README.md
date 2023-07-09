# MNIST Neural Network

This script is a basic implementation of a Neural Network in Python using NumPy from scratch. It's designed to classify handwritten digits from the MNIST dataset.

## Features

- Loads the MNIST dataset from local files.
- Preprocesses the data (normalizes pixel values and one-hot encodes labels).
- Trains a neural network on the dataset using sigmoid activation and stochastic gradient descent.
- Evaluates the trained model's performance on a test set.
- Saves a plot of the model's accuracy over training epochs.

## Requirements

- Python 3
- NumPy
- Matplotlib

## Usage

1. Update `labels_file_path` and `images_file_path` with your local paths to the MNIST dataset.
2. Run the script with Python: `python mnist_cnn.py`.

## Outputs

- Console output of model accuracy every 10 epochs and final test accuracy.
- A 'accuracy.png' plot file showing model accuracy over epochs.
