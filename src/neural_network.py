import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Read the data
data = pd.read_csv('data/train.csv')
# Turn panda data into numpy array
data = np.array(data)

# Shuffle the data
m, n = data.shape
np.random.shuffle(data)

# Split data into development and testing sets
# Developing set: 1000 samples
development_data = data[0:1000].T
development_data_labels = development_data[0]
development_data_pixels = development_data[1:n]
# Testing set: 1000 samples
training_data = data[1000:m].T
training_data_labels = training_data[0]
training_data_pixels = training_data[1:n]

def init_params():
    """
    Initialize weights and bias.
    """
    weight_1 = np.random.rand(10, 784) - 0.5
    bias_1 = np.random.rand(10, 1) - 0.5
    weight_2 = np.random.rand(10, 784) - 0.5
    bias_2 = np.random.rand(10, 1) - 0.5
    return weight_1, bias_1, weight_2, bias_2

def ReLU(x):
    """
    ReLU activation function.

    Args:
        x - Input number
    Returns:
        0 if x < 0, x otherwise
    """
    return np.maximum(x, 0)

def softmax(x):
    """
    Softmax activation function.

    Args:
        x - Input number
    Returns:
        Softmax function value
    """
    return np.exp(x) / np.sum(np.exp(x))

def forward(w1, b1, w2, b2, d):
    """
    Forward propagation.

    Args:
        - w1 - Weights of the first layer
        - b1 - Bias of the first layer
        - w2 - Weights of the second layer
        - b2 - Bias of the second layer
        - d  - Input data
    Returns:
        - z1 - Linear transformation output of the first layer
        - a1 - Activation output of the first layer
        - z2 - Linear transformation output of the second layer
        - a2 - Activation output of the second layer
    """
    z1 = w1.dot(d) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

def one_hot_encode(y):
    """
    One hot encode the labels.

    :param y: Labels
    :return: One hot encoded labels
    """
    encoded = np.zeros((y.size, y.max() + 1))
    encoded[np.arange(y.size), y] = 1
    return encoded.T

def backward(z1, a1, z2, a2, w2, l):
    """
    Backward propagation.
    """
    pass