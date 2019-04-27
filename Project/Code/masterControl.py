"""
Created on Sat Apr 27 11:44:13 2019

@author: Aswin Matthews Ashok

"""

import applicationInterface as app
import cnnInterface as cnn
import dataInterface as data

if __name__ == '__main__':
	"""Runs the entire application"""
	cnn.loadModel()
	print("====================Main Interface====================")
	while True:
		print("\nOptions:\n1. Run Applications.\n2. Process Data for training.\n3. Build and test network.\n4. Exit")
		opt = input("Enter the option number : ")
		if opt == "1":
			app.interface()
		elif opt == "2":
			data.interface()
		elif opt == "3":
			cnn.interface()
		elif opt == "4":
			break
		else:
			print("Invalid option")