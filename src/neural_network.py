import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

'''
The network is a simple 2-layer neural network with ReLU activation function for the hidden layer and softmax activation function for the output layer.
The input layer has 784 units, the hidden layer has 10 units, and the output layer has 10 units.

The network is trained using the training data and labels. The training data is a 784x1000 matrix, where each column is a flattened image of size 28x28.
The training labels is a 1000x1 vector, where each element is the label of the corresponding image in the training data.
'''

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
    weight_1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784)) * 0.05
    bias_1   = np.random.normal(size=(10, 1)) * np.sqrt(1./(10))    * 0.05
    weight_2 = np.random.normal(size=(10, 10)) * np.sqrt(1./(20))   * 0.05
    bias_2   = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))   * 0.05
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

def dReLU(x):
    """
    Derivative of ReLU activation function.

    Args:
        x - Input number
    Returns:
        0 if x < 0, 1 otherwise
    """
    return x > 0

def softmax(x):
    """
    Softmax activation function.

    Args:
        x - Input number
    Returns:
        Softmax function value
    """
    # Ensure input values are finite
    x = np.clip(x, -1e10, 1e10)
    
    exp_shifted = np.exp(x - np.max(x))
    sum_exp_shifted = np.sum(exp_shifted, axis=0, keepdims=True)
    # Add a small epsilon to avoid division by zero
    return exp_shifted / (sum_exp_shifted + 1e-10)

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

    Args:
        - y - Labels
    Returns:
        - One hot encoded labels
    """
    encoded = np.zeros((y.size, y.max() + 1))
    encoded[np.arange(y.size), y] = 1
    return encoded.T

def backward(z1, a1, z2, a2, w2, d, l):
    """
    Backward propagation.

    Args:
        - z1  - Linear transformation output of the first layer
        - a1  - Activation output of the first layer
        - z2  - Linear transformation output of the second layer
        - a2  - Activation output of the second layer
        - w2  - Weights of the second layer
        - d   - Data
        - l   - Labels
    Returns:
        - dw1 - Gradient of the weights of the first layer
        - db1 - Gradient of the bias of the first layer
        - dw2 - Gradient of the weights of the second layer
        - db2 - Gradient of the bias of the second layer
    """
    one_hot_labels = one_hot_encode(l)
    m = l.size
    dz2 = a2 - one_hot_labels
    dw2 = 1/m * dz2.dot(a1.T)
    db2 = 1/m * np.sum(dz2, 1, keepdims=True)
    dz1 = w2.T.dot(dz2) * dReLU(z1)
    dw1 = 1/m * dz1.dot(d.T)
    db1 = 1/m * np.sum(dz1, 1, keepdims=True)
    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    """
    Update the weights and bias.

    Args:
        - w1    - Weights of the first layer
        - b1    - Bias of the first layer
        - w2    - Weights of the second layer
        - b2    - Bias of the second layer
        - dw1   - Gradient of the weights of the first layer
        - db1   - Gradient of the bias of the first layer
        - dw2   - Gradient of the weights of the second layer
        - db2   - Gradient of the bias of the second layer
        - alpha - Learning rate
    Returns:
        - Updated weights and bias
    """
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

def gradient_descent(d, l, iterations, alpha):
    """
    Perform gradient descent.

    Args:
        - d             - Data
        - l             - Labels
        - iterations    - Number of iterations
        - alpha         - Learning rate
    Returns:
        - Trained weights and bias
    """
    w1, b1, w2, b2 = init_params()
    for iteration in range(iterations):
        z1, a1, z2, a2 = forward(w1, b1, w2, b2, d)
        dw1, db1, dw2, db2 = backward(z1, a1, z2, a2, w2, d, l)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if iteration % 10 == 0:
            print(f"Iteration: {iteration} - Loss: {get_loss(a2, l):.10f} - Accuracy: {get_accuracy(l, get_predictions(a2)):.3f}")
        if  abs(get_accuracy(l, get_predictions(a2)) - 1) < 0.03:
            print(f"Final iteration: {iteration} - Loss: {get_loss(a2, l):.10f} - Accuracy: {get_accuracy(l, get_predictions(a2)):.3f}")
            break
    return w1, b1, w2, b2

def get_predictions(a2):
    """
    Get predictions.

    Args:
        - a2 - Activation output of the second layer
    Returns:
        - Predictions
    """
    return np.argmax(a2, 0)

def get_accuracy(l, predictions):
    """
    Get accuracy.

    Args:
        - l           - Labels
        - predictions - Predictions
    Returns:
        - Accuracy
    """
    return np.sum(predictions == l) / l.size

def get_loss(a2, l):
    """
    Get loss.

    Args:
        - a2 - Activation output of the second layer
        - l  - Labels
    Returns:
        - Loss
    """
    epsilon = 1e-10  # Small value to avoid log(0)
    one_hot_labels = one_hot_encode(l)
    return np.sum(-np.log(a2 + epsilon) * one_hot_labels)

def make_predictions(w1, b1, w2, b2, d):
    """
    Make predictions.

    Args:
        - w1 - Weights of the first layer
        - b1 - Bias of the first layer
        - w2 - Weights of the second layer
        - b2 - Bias of the second layer
        - d  - Data
    Returns:
        - Predictions
    """
    _, _, _, a2 = forward(w1, b1, w2, b2, d)
    return get_predictions(a2)

def test_predictions(index, w1, b1, w2, b2):
    """
    Test the predictions.

    Args:
        - index - Index of the data
        - w1    - Weights of the first layer
        - b1    - Bias of the first layer
        - w2    - Weights of the second layer
        - b2    - Bias of the second layer
    """
    current_image = training_data_pixels[:, index, None]
    prediction = make_predictions(w1, b1, w2, b2, current_image)
    label = training_data_labels[index]
    print(f"Prediction: {prediction} - Label: {label}")

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()