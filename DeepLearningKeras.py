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

scaled_train_samples, train_labels = [], []

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

""" Underfitting
Underfitting describes a situation where the model is unable to predict the outcome of any data.
This can be solved by making the data more complex or the model more complex. This can include reducting
dropout.
"""

""" Regularization
Regularization is a technique to reduce overfitting and variance in the network, by penilizing for 
complexity. This can be done by adding a term to the loss function, penilizing for large weights.
"""

from keras import regularizers

model3 = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu', kernel_regulatizer=regularizers.l2(0.01)), # Specifies use of the L2 regularizer, and 0.01 as the regularization parameter. A constant that has to be tweeked for the best use.
    Dense(2, activation='sigmoid')
])

""" Learning rate
When the weights in the layers of the neural network is updated, it is done by changing the weights
based on the gradient denoted by: the loss / the weight. This gradient is multiplied by the learningrate.
A large learningrate will result in bigger changes each update. A normal learning rate is between 0.01
and 0.0001. Can be compaired to newtons method.
"""
model2.optimizer.lr = 0.01

""" Batch size
The batch size is the number of samples that will be passed through the network at one time. Larger batch
size will make the model faster to train. This occures because the computer can process the data in
parallel. This can however degrade the accuracy of the prediction because: One batch will account for one
update of the neural network based on the average of the loss function for the given data. This can 
reduce variance but also make the training to general.
"""

""" Fine-tuning
Fine-tuning is the process of utilizing a model trained for a specific purpose, and tune it to perform
a different but similar task. Hence reusing prior work and reduce need for computational power.
This can be done by removing the last layers and freezing the prior layers while training the new layers.
"""

""" Data augmentation
Data augmentation is a process that changes the data in a way that in theory should not change the output.
Examples of this would be to rotate pictures. The content would be the same, but it can increase the data set.
"""

""" Predicting
When the model is trained it can be used to predict based on test data. The test data will not provide
the desired output to the network.
"""

scaled_test_samples = []

predictions = model3.predict(scaled_test_samples, batch_size=10, verbose=0) # Verbose will determine the amount of data from the prediction that is printed to the terminal.

""" Supervised learning
Supervised learning is denoted by the data it trains on includes the lable (wanted output).
Underneath is an example of how supervised learning can be used in keras.
"""

# weight, height
train_samples = [[150, 67], [130, 60], [200, 65], [125, 52], [230, 72], [181, 70]]

# 0: male, 1: female
train_labels = [1, 1, 0, 1, 0, 0]

model3.fit(x=train_samples, y=train_labels, batch_size=3, shuffle=True, verbose=2)

""" Unsupervised learning
Unsupervised learning is characterized by the training data not having lables. There is therefor no
way of looking at accuracy, since it is not defined. The training will rather try to structure the data.
The model can cluster the data and learn the structure. Weather the clusters have anything in common in
the real world is outside of the models scope.
An autoencoder takes in input and sends outputs a reconstruction of the given input. An application can
be to reconstruct images with noise to only encompas the meaningfull part that will be utilized further.
"""

""" Semi-supervised learning
Semi-supervised learning uses both labled and unlabled data. First the model is trained supervised.
Then the unlabeled data is labeled by the model (called pseudo-labeling) and the entire dataset is 
used for further training.
"""

""" Convolutional neural networks (CCN)
CCN are widely used for image analasys. CNNs are good for patern recognition. CNN has convolutional layers
as its hiden layers. A convolutional layer differs from other layers by using convolution to calculate
the node values. Each layer has a number of filters, matricies, used to calculate the value of a node in
layer. Given that the input to the NN is a 2d list, the matrix will be slided over, some computation will
occure between the two matricies, f.eks a dot product, and the product will be the value of one node.
The matricies used as filters can have any value to recognize any desired patern.
The layers detects paterns, and many layers can detect more complex shapes hence making
it optimal for image recongition.
For visual explanation see: https://www.youtube.com/watch?v=YRhxdVk_sIs&list=PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU&index=21
"""

# Video 22 shows a visualization of the filters of a convolutional neural network.

""" One-hot encodings
One-hot encoding is a way of encoding the labels of the data. When training a NN to understand predict
pictures of cats and dogs, the NN will not undestand these words. Instead we can use a vector where one
spot marks one of the classifications the NN is to classify. Dogs: [1,0], Cats: [0,1]
One-hot referes to the system where only one positon can be active at a time.
"""

""" Batch normalization
Normalization aims to map data into the same data scale. eks: between 0 and 1. This can minimize occations
where values in the NN becomes so large that they dominate the other weights and values. Inputs should
therefor be normalized.
Batch normalization is the process of normalizing weights and activationfunctions inside the layers of
the neural network.
"""

from keras.layers import BatchNormalization

model4 = Sequential([
    Dense(16, input_shape=(1,), activation='relu'),
    Dense(32, activation='relu'),
    BatchNormalization(axis=1), # Follows the layer to be normalized. Specifies that axis 1 should be normalized.
    Dense(2, activation='softmax')
])

# beta_initializer: Initializer for the beta weight.
# gamma_initializer: Initializer for the gamma weight.

""" Zero padding
When using convolution by filters the resulting matrix will be smaller than the original input.
nxn image
fxf filter
output size = (n-f+1)x(n-f+1)
This is a problem because the image will undergo this change for every layer, giving the posibility to
end up with very few values and hence loosing data.
Zero padding adds zerovalues for surounding pixels in the input to make the output the same dimintion as
the input.

Types of padding:
valid - no padding.
same - pading to make the output the same size as the input size.
"""

from keras.layers.convolutional import *
from keras.layers import Flatten

model_valid = Sequential([
    Dense(16, activation='relu', input_shape=(20,20,3)),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='valid'),  # kernel_size is the size of the filter.
    Conv2D(64, kernel_size=(5, 5), activation='relu', padding='valid'),
    Conv2D(128, kernel_size=(7, 7), activation='relu', padding='valid'),
    Flatten(),
    Dense(2, activation='softmax')
]) 

model_valid.summary() # Prints the layer structure, with type, output shape, paramaters.

model_same = Sequential([
    Dense(16, activation='relu', input_shape=(20,20,3)),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),  # kernel_size is the size of the filter.
    Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
    Conv2D(128, kernel_size=(7, 7), activation='relu', padding='same'),
    Flatten(),
    Dense(2, activation='softmax')
]) 

model_same.summary()

""" Max pooling
Max pooling is added after a convolution layer and decreases the amount of outputs. Max pooling uses a
filter as in zero padding, and a stride to determine the amount of positions in the matrix to move per
time the filter is applied. The max value of the values fitted into the filter will be the output for
the nodes covered by the filter.
Why: reduces computational cost and can hence look at lager parts of the image at the same time.
The theory behind the method are that the max value are the most activated once, and is therefor most interesting.
"""
from keras.layers.pooling import *

model5 = Sequential([
    Dense(16, activation='relu', input_shape=(20,20,3)),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),       # Performs a max pooling on the concolution layer with 32 nodes.
    Conv2D(64, kernel_size=(7, 7), activation='relu', padding='same'),
    Flatten(),
    Dense(2, activation='softmax')
]) 


#******************************************************
#---------------Part  3 - Backpropagation--------------
#******************************************************

""" Process of backpropagation
1 Data is passed to the neural network and it makes a predoction.
2 The loss is calculated.
3 The optimization alorithm, f.eks. gradient decendt, is used to update the network to minimize the loss.
    This is done by calculating the gradient of the loss with respect to the weights
"""

# Continue on video 28 Backproparagtion explained Part 2 - The mathematical notation
