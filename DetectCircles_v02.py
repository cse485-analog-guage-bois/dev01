import cv2 
import numpy as np

img = cv2.imread("circles2.jpg") # Read jpg image
output = img.copy() # make a copy of the original image
blur = cv2.medianBlur(img,13) # blur the original image
gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY) # convert the blurred image into gray scale
edges = cv2.Canny(gray,50,150,apertureSize = 3) # Find the edges from the gray scale copy of image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100) # Detect Circles from gray scale
 
if circles is not None: # ensure that some circles were found
	circles = np.round(circles[0, :]).astype("int") # make x and y coordinates and radius of circle integer values
	for (x, y, r) in circles: # cycle through all circles found
		cv2.circle(output, (x, y), r, (0, 255, 0), 4) # draw circles around each circle
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1) # draw rectangle in center of each circle
 
	cv2.imwrite("output_v01.jpg", output) # ouput the drawn circles image to specified jpg format