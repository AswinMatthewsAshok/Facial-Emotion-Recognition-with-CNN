"""
Created on Sat Apr 27 11:44:13 2019

@author: Aswin Matthews Ashok

"""

import numpy as np
import cv2
import cnnInterface as cnn
import faceDetectorInterface as d
import keras as k
import os
import mss

VIDEO_PATH = "../Dataset/video.mp4"
EMOTION = ["0_NEUTRAL","1_ANGER","2_CONTEMPT","3_DISGUST","4_FEAR","5_HAPPY","6_SADNESS","7_SURPRISE"]

def outputProcessing(frame,x,y,w,h,categorical):
	"""Formats the output image to show predicted labels"""
	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
	color = [(255,0,0) for x in categorical]
	color[np.argmax(categorical)] = (0,255,0)
	cv2.putText(frame, EMOTION[0]+" "+str(int(categorical[0]*100)), (x,y+h+16), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[0],1, cv2.LINE_AA)
	cv2.putText(frame, EMOTION[1]+" "+str(int(categorical[1]*100)), (x,y+h+32), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[1],1, cv2.LINE_AA)
	cv2.putText(frame, EMOTION[2]+" "+str(int(categorical[2]*100)), (x,y+h+48), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[2],1, cv2.LINE_AA)
	cv2.putText(frame, EMOTION[3]+" "+str(int(categorical[3]*100)), (x,y+h+64), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[3],1, cv2.LINE_AA)
	cv2.putText(frame, EMOTION[4]+" "+str(int(categorical[4]*100)), (x,y+h+80), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[4],1, cv2.LINE_AA)
	cv2.putText(frame, EMOTION[5]+" "+str(int(categorical[5]*100)), (x,y+h+96), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[5],1, cv2.LINE_AA)
	cv2.putText(frame, EMOTION[6]+" "+str(int(categorical[6]*100)), (x,y+h+112), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[6],1, cv2.LINE_AA)
	cv2.putText(frame, EMOTION[7]+" "+str(int(categorical[7]*100)), (x,y+h+128), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[7],1, cv2.LINE_AA)

def detectFacesInWebcam():
	"""Detects facial emotions in webcam feed"""
	if cnn.loadModel():
		cap = cv2.VideoCapture(0)
		while(cap.isOpened()):
			ret, frame = cap.read()
			coordinates,gray = d.detectFace(frame,True)
			for (x,y,w,h) in coordinates:
				face = cv2.resize(gray[y:y+h,x:x+w],d.FACE_DIMENSIONS)
				outputProcessing(frame,x,y,w,h,cnn.predictImageLables(face)[0])
			cv2.imshow('frame',frame)
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		cap.release()
		cv2.destroyAllWindows()
	else:
		print("Invalid Model path "+cnn.MODEL_PATH)

def detectFacesInVideo(videoPath):
	"""Plays the video in the path with facial emotion labels"""
	if cnn.loadModel():
		if os.path.exists(videoPath):
			cap = cv2.VideoCapture(videoPath)
			while(True):
				ret, frame = cap.read()
				if ret:
					coordinates,gray = d.detectFace(frame,True)
					for (x,y,w,h) in coordinates:
						face = cv2.resize(gray[y:y+h,x:x+w],d.FACE_DIMENSIONS)
						outputProcessing(frame,x,y,w,h,cnn.predictImageLables(face)[0])
					cv2.imshow('frame',frame)
					if cv2.waitKey(25) & 0xFF == ord('q'):
						break
				else:
					break
			cap.release()
			cv2.destroyAllWindows()
		else:
			print("Invalid video path "+videoPath)
	else:
		print("Invalid Model path "+cnn.MODEL_PATH)

def detectFacesOnScreen():
	"""Detects facial emotions of faces on the computer screen"""
	if cnn.loadModel():
		with mss.mss() as sct:
			monitor = {'top': 0, 'left': 0, 'width': 800, 'height': 900}
			while True:
				frame = np.array(sct.grab(monitor))
				coordinates,gray = d.detectFace(frame,True)
				for (x,y,w,h) in coordinates:
					face = cv2.resize(gray[y:y+h,x:x+w],d.FACE_DIMENSIONS)
					outputProcessing(frame,x,y,w,h,cnn.predictImageLables(face)[0])
				cv2.imshow('frame', frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
	else:
		print("Invalid Model path "+cnn.MODEL_PATH)

def interface():
	print("====================Application Interface====================")
	while True:
		print("\nOptions:\n1. Detect facial emotions in webcam.\n2. Detect facial emotions in video.\n3. Detect facial emotions on screen.\n4. Exit")
		opt = input("Enter the option number : ")
		if opt == "1":
			detectFacesInWebcam()
		elif opt == "2":
			detectFacesInVideo(VIDEO_PATH)
		elif opt == "3":
			detectFacesOnScreen()
		elif opt == "4":
			break
		else:
			print("Invalid option")
	print("\n====================Main Interface====================")