"""
Created on Sat Apr 27 11:44:13 2019

@author: Aswin Matthews Ashok

"""

import numpy as np
import keras as k
import cv2

def trainCnnNetwork(trainX, trainY, valX, valY, convLayers, denseLayers, epochs, storagePath):
	trainY = k.utils.to_categorical(trainY)
	valY = k.utils.to_categorical(valY)
	# Model Setup
	model = k.models.Sequential()
	layerCount = 0
	model.add(k.layers.BatchNormalization())
	for (filters,kerel_size,activation) in convLayers:
		layerCount+=1
		if layerCount==1:
			model.add(k.layers.Conv2D(filters, kerel_size, padding = "same", activation = activation,input_shape = trainX.shape[1:]))
		else:
			model.add(k.layers.Conv2D(filters, kerel_size, padding = "same", activation = activation))
		model.add(k.layers.MaxPooling2D(pool_size=(2, 2)))
		model.add(k.layers.BatchNormalization())
		model.add(k.layers.Dropout(0.1))
	model.add(k.layers.Flatten())
	for (neurons,activation) in denseLayers:
		layerCount+=1
		model.add(k.layers.Dense(neurons, activation = activation))
		if(activation!='softmax'):
			model.add(k.layers.BatchNormalization())
			model.add(k.layers.Dropout(0.1))
	# Model Compilation
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	# Model Training
	history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs = epochs, batch_size = 32)
	# Print model summary
	model.summary()
	print(history.history.keys())
	# Storing Model
	model.save(storagePath)

def predictArrayLabels(data,model):
	categorical = model.predict(data)
	predictions = np.argmax(categorical,axis = 1)
	return predictions

def predictImageLabel(image,model):
	image = np.expand_dims(np.array([image]),axis = 3)
	data = image.astype(float)
	categorical = model.predict(data)
	return categorical