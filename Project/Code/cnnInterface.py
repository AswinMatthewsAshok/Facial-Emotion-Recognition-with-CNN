"""
Created on Sat Apr 27 11:44:13 2019

@author: Aswin Matthews Ashok

"""

import numpy as np
import dataInterface as d
from  sklearn.model_selection import train_test_split
import sklearn.metrics as m
import emotionRecognitionNetwork as e
import os
import keras as k

TRAIN_PERCENT = 80
VALIDATION_PERCENT = 10
MODEL_PATH = "../Support/model.h5"
EPOCHS = 5


MODEL = False

def loadModel():
	"""Load existing model"""
	global MODEL
	if(MODEL):
		return True
	else:
		if os.path.exists(MODEL_PATH):
			MODEL = k.models.load_model(MODEL_PATH)
			return True
		else:
			return False

def splitDataTrainValidTest(trainPercent,valPercent):
	"""Split data into train, test, validation sets"""
	global trainX,valX,testX,trainY,valY,testY
	trainX,valX,trainY,valY = train_test_split(data,labels,test_size = (100-trainPercent)/100,random_state = 42)
	testPercent = 100 - (trainPercent+valPercent)
	valX,testX,valY,testY = train_test_split(valX,valY,test_size = testPercent/(valPercent+testPercent),random_state = 42)
	return trainX,valX,testX,trainY,valY,testY

def trainModel():
	"""Configures network structure and trains the network"""
	global trainX,valX,testX,trainY,valY,testY
	trainX,valX,testX,trainY,valY,testY = splitDataTrainValidTest(TRAIN_PERCENT,VALIDATION_PERCENT)

	convLayers= [
	(32,3,'relu'),
	(64,3,'relu'),
	(128,3,'relu'),
	]

	denseLayers = [
	(20,'relu'),
	(8,'softmax')
	]

	e.trainCnnNetwork(trainX,trainY,valX,valY,convLayers, denseLayers, EPOCHS, MODEL_PATH)

def predictTestLabels():
	"""Evaluates performance on test data"""
	predictions = predictArrayLabels(testX)
	if predictions[0]:
		predictions = predictions[1]
		expected = testY.flatten()
		print("Accuracy:",m.accuracy_score(expected,predictions))
		print("Confusion Matrix:\n",m.confusion_matrix(expected,predictions))

def predictImageLables(image):
	"""Predicts labels for a single image"""
	if loadModel():
		return e.predictImageLabel(image,MODEL)
	else:
		print("Invalid Model path "+MODEL_PATH)

def predictArrayLabels(data):
	"""Predict labels for an array of images"""
	if loadModel():
		return True,e.predictArrayLabels(data,MODEL)
	else:
		print("Invalid Model path "+MODEL_PATH)
		return False, None

def interface():
	"""Builds and tests the network"""
	global data,labels
	if os.path.exists(d.OUTPUT_DIRECTORY+"/data.npy") and os.path.exists(d.OUTPUT_DIRECTORY+"/labels.npy"):
		data = np.expand_dims(np.load(d.OUTPUT_DIRECTORY+"/data.npy"),axis = 3)
		labels = np.expand_dims(np.load(d.OUTPUT_DIRECTORY+"/labels.npy"),axis = 1)
		trainModel()
		predictTestLabels()
	elif os.path.exists(d.OUTPUT_DIRECTORY+"/data.npy"):
		print("Invalid path "+d.OUTPUT_DIRECTORY+"/labels.npy")
	else:
		print("Invalid path "+d.OUTPUT_DIRECTORY+"/labels.npy")
	print("\n====================Main Interface====================")
