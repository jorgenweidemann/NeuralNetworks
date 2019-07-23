"""
This is a coding example explaining keras in depth.
It does however assume a basic prior knowlege of neural networks.
The example is based on a youtube tutorial by deeplizard
https://www.youtube.com/watch?v=gZmobeGL0Yg&list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU&index=1
"""

import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

#******************************************************
#-----------------------Part 1-------------------------
#******************************************************

# A sequential model is a has a linear stack of layers. 
# This is what is considered the standard neural network.
model1 = Sequential([ 
    
    # A dense layer is a standard layer that is fully connected, all inputs are connected to all outputs.
    Dense(5, input_shape = (10,), activation = 'relu'), # input_shape sets the format of the input layer and creates as an additional layer.
    Dense(2, activation = 'softmax') # The activation denotes a function to transform an input to an output which becomes the value of a given node in the network.
                                     # This can be utilized to give values of many sorts
])

"""
Activation functions:
sigmoid - maps any number to a number between 0 and 1
relu - Gives zero if the input is negative and the input if it is positive.
"""

# Alternatively:
identicalModel1 = Sequential()
identicalModel1.add(Dense(5, input_shape = (3,)))
identicalModel1.add(Activation('relu'))
identicalModel1.add(Dense(2, activation = 'sigmoid'))

""" Training
The training of a neural network is an optimization problem of the weights. This can be done by
optimizationer as
A valuation of the training that has taken place is the loss function. It measures the discrepnacy
between the actual output and the desired output.
"""

#******************************************************
#-----------------------Part 2-------------------------
#******************************************************


model2 = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model2.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
"""
For the neural network to train it needs to be compiled.
Adam(lr=0.0001) is the specification of an optimizer, here called Adam, and a learningrate of 0.0001

loss denotes the loss function used for the network.
Mean square error (mse) is a loss function and it takes the average of the squared difference between
output and true (wished) output.
"""

model2.fit(scaled_train_samples, train_labels, batch_size=10, epochs=20, shuffle=True, verbose=2)
"""
batch_size relates to the amount of data sent to the model at once.
epochs are the amount of iterations of training occuring on the given dataset.
shuffle will randomize data before it is used to train.
verbose determines the amount of prints.
"""

""" Data handling
When training a neural network we want to split our data in three categories:
1 Training data - This data will be used to train and has to contian the "optimal" output.
2 Validation data - This data is used to find accuracy and loss after an epoch of training. It has to
    contain the "optimal" output.
3 Test data - This is used after the neural network is finished training to make shure it is able to
    predict outputs without the answer. This does not need to contain any form of output.
"""

model2.fit(scaled_train_samples, train_labels, validation_split = 0.20, batch_size=10, epochs=20,
            shuffle=True, verbose=2)
"""
validation_split will here use 20% of the data to validate, rather than train.
validation_data = valid_set would be a way to manually separate traning data and validation data.
"""

""" Overfitting
Overfitting occures when the model gets very good at predicting traing data, but no validation data.
This means the model is unable to generalize the training to new situations.
Easiest fix is to add more data. Data augmentation can be used to modify existing data to expand the 
data set. 
Another way would be to decrease the amount of weights that needs to be fitted, by decreasing the amount
of neurons. This can be done my droput, which neutralises some nodes during training.
"""

# Continue on video 11 Underfitting