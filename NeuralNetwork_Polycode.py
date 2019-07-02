
import numpy as np

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3,1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        return x * (1 - x)

    def train(self, training_input, training_output):
        output = self.think(training_input)
        error = training_output - output
        adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
        self.synaptic_weights += adjustments
