"""
This file contains an attempt to make a neural network for use in a reinforcement learning project for NorgesGruppen.
The model is a simple binary not function.
"""

#************************************
#            Initial setup
#************************************

# Imports
import keras
import tensorflow
tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)
import numpy
import random

# Libary version check
print("You are running this program on:")
print("keras version: ", keras.__version__)
print("tensorflow version: ", tensorflow.VERSION)
print("numpy version: ", numpy.version.version)
print("\nDeveloped on:")
print("keras version:  2.2.4\ntensorflow:  1.14.0\nnumpy version:  1.16.1")

# Static variables
AREAHEIGHT = 4
AREAWIDTH = 4
TRAININGBATCHSIZE = 10
TRAININGEPOCHS = 10
TRAININGVERBOSE = 2
TRAININGINPUTS = 1_000
SAVELOADTHRESHOLD = 0.95
ALLWAYSTRAIN = False

# Dependet variables
inputSize = AREAHEIGHT * AREAWIDTH
outputSize = AREAHEIGHT * AREAWIDTH # Will always be the same as inputSize


#*******************************
#            Classes
#*******************************

# Class for my neural network.
class NeuralNetwork():

    # Initial class set-up. Constructor.
    def __init__(self):

        # Create the neural network.
        self.__NN = keras.models.Sequential([
            keras.layers.core.Dense(int(inputSize), input_dim=(int(inputSize))),
            keras.layers.core.Dense(inputSize),
            keras.layers.core.Dense(int((inputSize + outputSize)/2)),
            keras.layers.core.Dense(int(outputSize))
        ])

        # The neural network has to be compiled before use.
        self.__NN.compile(keras.optimizers.Adam(lr=0.001), loss='mse') # Can consider using epsilon and decal in the compilation.

        self.__accuracy = 0

    # Method for training the neural network for large data existing datasets.
    def train(self, setSize):
        
        # If an old model exists and the variable ALWAYSTRAIN=False it loads rather than train.
        loaded = False
        if not ALLWAYSTRAIN:
            try:
                self.load()
                loaded = True
                print("Model was loaded.")
            except IOError:
                print('Loading of old model failed.')
        if not loaded:

            # Creates data.
            trainingInput, trainingOutput = self.createTrainingData(setSize)

            # Use the standard training method that is used for supervised learning for this type of training.
            self.__NN.fit(trainingInput, trainingOutput, batch_size=TRAININGBATCHSIZE, epochs=TRAININGEPOCHS, shuffle=True, verbose=TRAININGVERBOSE)

            # Save the accuracy
            self.__accuracy = self.findAccuracy(100)

            print("Model was trained.")


    # Method to make a prediction of a good placement of a carrier. The main purpose is to predict one location for one input. It will work for more.
    def predict(self, input, listRepresentation=False):

        # Use the standard method for prediction in kreas.
        predictionList = self.__NN.predict(input, batch_size=None)
        if listRepresentation:
            return predictionList

        # Use numpy to find the coordinates.
        listLocation = numpy.argmax(predictionList)
        width = listLocation % AREAWIDTH
        height = (listLocation - width) / AREAWIDTH
        return numpy.array([int(width), int(height)])


    # Creates data for training.
    def createTrainingData(self, setSize):

        # Create trainingInput
        choices = [int(0),int(1)]
        inp = []
        for i in range(setSize):
            for r in range(inputSize):
                inp.append(random.choice(choices))
        inputData = numpy.reshape(inp, (setSize, inputSize))

        # Create trainingOutput
        out = []
        for i in range(setSize):
            for r in range(inputSize):
                if inputData[i][r] == 0:
                    out.append(1)
                else:
                    out.append(0)
        outputData = numpy.reshape(out, (setSize, inputSize))
        return inputData, outputData
    
    # Finds the accuracy
    def findAccuracy(self, testSetSize=100):
        inputForEvaluation, outputForEvaluation = self.createTrainingData(testSetSize)
        accuracy = 0
        for i in range(100):
            prediction = network.predict(inputForEvaluation[i].reshape((1,-1)), True)
            if outputForEvaluation[i][numpy.argmax(prediction)] == 1:
                accuracy += 1
        return accuracy / testSetSize


    # Method for saving model given it has an accuracy above the threshold.
    def save(self):
        if self.__accuracy > SAVELOADTHRESHOLD:
            self.__NN.save('NNForRL.model')
            print("Model was saved.")
        else:
            print("Model was not saved.")
    
    
    # Method for loading the last model that was saved.
    def load(self):
        self.__NN = keras.models.load_model('NNForRL.model')
        self.__accuracy = self.findAccuracy()

    
    # Getter for accuracy.
    def getAccuracy(self):
        return self.__accuracy


#*********************************
#              Test
#*********************************


if __name__ == "__main__":
    
    network = NeuralNetwork()

    network.train(TRAININGINPUTS)
    
    print("Accuracy: ", network.getAccuracy())

    prediction = network.predict(numpy.array([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,0]).reshape(1,-1))

    print("Predicted: ", prediction, " Expected: [3 3]")

    network.save()