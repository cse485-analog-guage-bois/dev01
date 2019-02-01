import cv2
import numpy as np

#img = cv2.imread("gauge-3.jpg")
img = cv2.imread("circles2.jpg")

output = img.copy()

blur = cv2.medianBlur(img,13)

# Convert the img to grayscale 
gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY) 
  
# Apply edge detection method on the image 
edges = cv2.Canny(gray,50,150,apertureSize = 3) 
  
# This returns an array of r and theta values 
lines = cv2.HoughLines(edges,1,np.pi/180, 200) 

# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
 
# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
 
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
	# show the output image
	cv2.imwrite("output.jpg", output)
      
# All the changes made in the input image are finally 
# written on a new image houghlines.jpg 
#cv2.imwrite('blur.jpg',blur)
#cv2.imwrite('BW.jpg', gray)
#cv2.imwrite('edges.jpg', edges)