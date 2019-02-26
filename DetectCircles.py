mport cv2 
import numpy as np

image = cv2.imread("circles2.jpg")

blurthis = cv2.medianBlur(image, 9)

edges = cv2.Canny(blurthis, 75, 100)

circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1.2, 100) #finds all circles based on edges

if circles is not None: 
	circleList = np.round(circles[0,:].astype("int")) #makes a list of circles

	for(x,y,r) in circles:
		cv2.circle(image, (x,y), r, (255,0,255), 5) #draws circle around circle with specified color and thickness on image

cv2.imshow("circles", image)