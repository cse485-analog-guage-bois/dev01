import cv2 
import numpy as np

image = cv2.imread("circles2.jpg")

blurthis = cv2.medianBlur(image, 13)

gray = cv2.cvtColor(blurthis, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100) #finds all circles based on edges

if circles is not None: 
	circleList = np.round(circles[0,:].astype("int")) #makes a list of circles

	for(x,y,r) in circles:
		cv2.circle(image, (x,y), r, (255,0,255), 5) #draws circle around circle with specified color and thickness on image

	cv2.imwrite("circles", image)

