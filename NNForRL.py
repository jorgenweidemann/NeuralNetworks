"""
This file contains an attempt to make a neural network for use in a reinforcement learning project for NorgesGruppen.
Disclamer: This is only an atempt and not exactly what will be implemented.
"""

# Imports
import keras
import tensorflow
import numpy
import random

print("keras version: ", keras.__version__)
print("tensorflow: ", tensorflow.VERSION)

# Static variables
AREAHEIGHT = 4
AREAWIDTH = 4
TRAININGBATCHSIZE = 10
TRAININGEPOCHS = 20
TRAININGVERBOSE = 2
TRAININGINPUTS = 10

# Dependet variables

inputSize = AREAHEIGHT * AREAWIDTH
outputSize = AREAHEIGHT * AREAWIDTH # Will always be the same as inputSize


# Class for my neural network.
class NeuralNetwork():

    # Initial class set-up. Constructor.
    def __init__(self):

        # Create the neural network.
        self.__NN = keras.models.Sequential([
            keras.layers.core.Dense(inputSize, input_dim=inputSize),
            keras.layers.core.Dense(inputSize),
            keras.layers.core.Dense((inputSize + outputSize)/2),
            keras.layers.core.Dense(outputSize, activation='softmax'),
        ])

        # The neural network has to be compiled before use.
        self.__NN.compile(keras.optimizers.Adam(lr=0.001)) # Can consider using epsilon and decal in the compilation.

    # Method for training the neural network for large data existing datasets.
    def train(self, trainingInput, trainingOutput):

        # Use the standard training method that is used for supervised learning for this type of training.
        self.__NN.fit(trainingInput, trainingOutput, batch_size=TRAININGBATCHSIZE, epochs=TRAININGEPOCHS, shuffle=True, verbose=TRAININGVERBOSE)

    # Method to make a prediction of a good placement of a carrier. The main purpose is to predict one location for one input. It will work for more.
    def predict(self, input):

        # Use the standard method for prediction in kreas.
        return self.__NN.predict(input, batch_size=None)


# Test


# Create trainingInput

choices = [int(0),int(1)]
values = []
for i in range(TRAININGINPUTS):
    for r in range(inputSize):
        values.append(random.choice(choices))
inputForTraining = numpy.reshape(values, (TRAININGINPUTS, inputSize))
print(inputForTraining)


# Create trainingOutput

values = []
for i in range(TRAININGINPUTS):
    for r in range(inputSize):
        if inputForTraining[i][r] == 0:
            values.append(1)
        else:
            values.append(0)
outputForTraining = numpy.reshape(values, (TRAININGINPUTS, inputSize))
print(outputForTraining)


# Test the network

network = NeuralNetwork()

# network.train(inputForTraining, outputForTraining)
