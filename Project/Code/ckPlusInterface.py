"""
Created on Sat Apr 27 11:44:13 2019

@author: Aswin Matthews Ashok

"""

import faceDetectorInterface as detector
import dataInterface as d
import cv2
import numpy as np
import os

LABELS_DIRECTORY = "../Dataset/CK+/Emotion_labels/Emotion"
IMAGES_DIRECTORY = "../Dataset/CK+/extended-cohn-kanade-images/cohn-kanade-images"
IMAGE_FILE_FORMAT = "png"

def storeFacesInEmotionDir():
	"""Navigates the CK+ directory structure, fetches the faces from the images and
	stores them in emotion directories according to their labels"""
	if os.path.exists(LABELS_DIRECTORY) and os.path.exists(IMAGES_DIRECTORY):
		for parent in os.listdir(LABELS_DIRECTORY):
			for child in os.listdir(LABELS_DIRECTORY+"/"+parent):
				emotion = 0
				labelFileNames = os.listdir(LABELS_DIRECTORY+"/"+parent+"/"+child)
				if len(labelFileNames)==1:
					labelFile = open(LABELS_DIRECTORY+"/"+parent+"/"+child+"/"+labelFileNames[0],"r")
					emotion = int(labelFile.readline()[3])
					labelFile.close()
				faces = detector.detectFacesInDir(IMAGES_DIRECTORY+"/"+parent+"/"+child,IMAGE_FILE_FORMAT)
				for (file,face) in faces:
					if not os.path.exists(d.OUTPUT_DIRECTORY+d.EMOTION_DIRECTORIES[emotion]+"/"+file):
						cv2.imwrite(d.OUTPUT_DIRECTORY+d.EMOTION_DIRECTORIES[emotion]+"/"+file,face)
		return True
	elif not os.path.exists(LABELS_DIRECTORY):
		print("Invalid path "+ LABELS_DIRECTORY)
	else:
		print("Invalid path "+ IMAGES_DIRECTORY)
	return False
