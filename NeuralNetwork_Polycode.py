
import numpy as np

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3,1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_input, training_output, iterations):
        for i in range(iterations):
            output = self.think(training_input)
            error = training_output - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

if __name__ == '__main__':
    neuralnetwork = NeuralNetwork()
    print('Random synaptic weights: ')
    print(neuralnetwork.synaptic_weights)

    training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    neuralnetwork.train(training_inputs, training_outputs, 1000)

    print('Synaptic weights after training: ')
    print(neuralnetwork.synaptic_weights)

    print('Outputs after training: ')
    print(neuralnetwork.think(training_inputs[0]),
        neuralnetwork.think(training_inputs[1]),
        neuralnetwork.think(training_inputs[2]),
        neuralnetwork.think(training_inputs[3]) 
        )
    
    print('Expected values:')
    print(training_outputs)

