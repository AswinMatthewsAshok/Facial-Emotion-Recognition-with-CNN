"""
Created on Sat Apr 27 11:44:13 2019

@author: Aswin Matthews Ashok

"""

import numpy as np
import cv2
import os

CLASSIFIER_PATH = "../Support/lbpcascade_frontalface_improved.xml"
FACE_DIMENSIONS = (100,100)

emotion = ["0_NEUTRAL","1_ANGER","2_CONTEMPT","3_DISGUST","4_FEAR","5_HAPPY","6_SADNESS","7_SURPRISE"]

def detectFace(image,isColor = True):
	"""Detects all faces in an image"""
	if os.path.exists(CLASSIFIER_PATH):
		if isColor:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		detector = cv2.CascadeClassifier(CLASSIFIER_PATH)
		coordinates = detector.detectMultiScale(image, 1.1, 2)
		for i in range(0,len(coordinates)):
			w = coordinates[i][2]
			h = coordinates[i][3]
			x = coordinates[i][0]
			y = coordinates[i][1]
			extraw = int(w*1/10)
			extrah = int(h*1/10)
			if(x-extraw >=0 and y-extrah >=0 and x+w+extraw < image.shape[1] and y+h+extrah < image.shape[0]):
				coordinates[i][0] = coordinates[i][0]-extraw
				coordinates[i][1] = coordinates[i][1]-extrah
				coordinates[i][2] = coordinates[i][2]+2*extraw
				coordinates[i][3] = coordinates[i][3]+2*extrah
		if isColor:
			return coordinates,image
		return coordinates
	else:
		print("Invalid path "+ CLASSIFIER_PATH)
		return [],[]

def detectFacesInDir(directory,fileType,):
	"""Detects faces in all images of a particular directory"""
	files = os.listdir(directory)
	output = []
	for file in files:
		typeLen = len(fileType)+1
		fileFormat = file[len(file)-typeLen:]
		if("."+fileType==fileFormat):
			image = cv2.imread(directory+"/"+file,cv2.IMREAD_GRAYSCALE)
			coordinates = detectFace(image,False)
			for (x,y,w,h) in coordinates:
				face = image[y:y+h,x:x+w]
				output.append((file,cv2.resize(face,FACE_DIMENSIONS)))
	return output