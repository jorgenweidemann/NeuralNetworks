""" NOT FINISHED
This file contains an attempt to make a neural network for use in a reinforcement learning project for NorgesGruppen.
Disclamer: This is only an atempt and not exactly what will be implemented.
"""

# Imports
import keras
import tensorflow
tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
import numpy
import random

print("keras version: ", keras.__version__)
print("tensorflow: ", tensorflow.VERSION)
print("numpy version: ", numpy.version.version)

# Static variables
AREAHEIGHT = 4
AREAWIDTH = 4
TRAININGBATCHSIZE = 10
TRAININGEPOCHS = 10
TRAININGVERBOSE = 2
TRAININGINPUTS = 400

# Dependet variables

inputSize = AREAHEIGHT * AREAWIDTH
outputSize = AREAHEIGHT * AREAWIDTH # Will always be the same as inputSize


# Class for my neural network.
class NeuralNetwork():

    # Initial class set-up. Constructor.
    def __init__(self):
        print(type(inputSize))

        # Create the neural network.
        self.__NN = keras.models.Sequential([
            keras.layers.core.Dense(int(inputSize), input_dim=(int(inputSize))),
            keras.layers.core.Dense(inputSize),
            keras.layers.core.Dense(int((inputSize + outputSize)/2)),
            keras.layers.core.Dense(int(outputSize))
        ])

        # The neural network has to be compiled before use.
        self.__NN.compile(keras.optimizers.Adam(lr=0.001), loss='mse', metrics=['accuracy']) # Can consider using epsilon and decal in the compilation.

    # Method for training the neural network for large data existing datasets.
    def train(self, trainingInput, trainingOutput):

        # Use the standard training method that is used for supervised learning for this type of training.
        self.__NN.fit(trainingInput, trainingOutput, batch_size=TRAININGBATCHSIZE, epochs=TRAININGEPOCHS, shuffle=True, verbose=TRAININGVERBOSE)

    # Method to make a prediction of a good placement of a carrier. The main purpose is to predict one location for one input. It will work for more.
    def predict(self, input):

        # Use the standard method for prediction in kreas.
        return self.__NN.predict(input, batch_size=None)


#*********************************
#              Test
#*********************************


# Create trainingInput
def createTrainingData(trainingInputs):
    choices = [int(0),int(1)]
    inp = []
    for i in range(trainingInputs):
        for r in range(inputSize):
            inp.append(random.choice(choices))
    inputData = numpy.reshape(inp, (trainingInputs, inputSize))

    # Create trainingOutput
    out = []
    for i in range(trainingInputs):
        for r in range(inputSize):
            if inputData[i][r] == 0:
                out.append(1)
            else:
                out.append(0)
    outputData = numpy.reshape(out, (trainingInputs, inputSize))
    return inputData, outputData


# Finds the accuracy
def findAccuracy(testSetSize):
    inputForEvaluation, outputForEvaluation = createTrainingData(testSetSize)
    accuracy = 0
    for i in range(100):
        prediction = network.predict(inputForEvaluation[i].reshape((1,-1)))
        if outputForEvaluation[i][numpy.argmax(prediction)] == 1:
            accuracy += 1
    return accuracy / testSetSize

if __name__ == "__main__":
    
    network = NeuralNetwork()

    inputForTraining, outputForTraining = createTrainingData(TRAININGINPUTS)

    network.train(inputForTraining, outputForTraining)
    
    print("Accuracy: ", findAccuracy(100))