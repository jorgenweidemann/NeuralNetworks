"""
Example from youtube video by sentdex
https://www.youtube.com/watch?v=wQ8BIBpya2k
"""

import tensorflow as tf 
print('TensorFlow version: ', tf.__version__)
mnist = tf.keras.datasets.mnist # 28x28 images fo hand-written digts 0-9
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

xTrain = tf.keras.utils.normalize(xTrain, axis = 1)
xTest = tf.keras.utils.normalize(xTest, axis = 1)

"""
import matplotlib.pyplot as plt 
print(xTrain[0])
plt.imshow(xTrain[0], cmap = plt.cm.binary)
plt.show()
"""

model = tf.keras.models.Sequential()

# Input layer
model.add(tf.keras.layers.Flatten()) 

# Hidden layer nr. 1 with recrified linear activation
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) 

# Hidden layer nr. 2
model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) 

# Output layer
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile( optimizer = 'adam',
               loss = 'sparse_categorical_crossentropy',
               metrics = ['accuracy'])
model.fit(xTrain, yTrain, epochs = 3)

# Evaluate validity of training
valLoss, valAcc = model.evaluate(xTest, yTest)
print(valLoss, valAcc)

# Try to predict
predictions = model.predict()

"""
# Save and load model
model.save('predictMnist.model')
oldModel = tf.keras.models.load_model('predictMnist.model')
predictions = oldModel.predict(xTest)
print('Predictions from old model:')
print(predictions)
"""