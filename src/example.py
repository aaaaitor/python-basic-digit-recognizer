import neural_network as nn

'''
Example of using the neural network module.
'''

# Gradient descent with 999 iterations and learning rate 0.001
weight1, bias1, weight2, bias2 = nn.gradient_descent(nn.development_data_pixels, nn.development_data_labels, 999, 0.001)

# Test prediction
nn.test_predictions(999, weight1, bias1, weight2, bias2)